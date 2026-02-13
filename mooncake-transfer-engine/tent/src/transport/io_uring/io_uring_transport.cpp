// Copyright 2024 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tent/transport/io_uring/io_uring_transport.h"

#include <cstdint>
#include <glog/logging.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <memory>

#include "tent/runtime/slab.h"
#include "tent/common/utils/os.h"
#include "tent/runtime/platform.h"

namespace mooncake {
namespace tent {
class IOUringFileContext {
   public:
    explicit IOUringFileContext(const std::string& path) : ready_(false) {
        fd_ = open(path.c_str(), O_RDWR | O_DIRECT);
        if (fd_ < 0) {
            PLOG(ERROR) << "O_DIRECT open failed for " << path
                        << " (O_DIRECT is required, no buffered fallback)";
            return;
        }
        LOG(INFO) << "File " << path << " opened with O_DIRECT";
        ready_ = true;
    }

    IOUringFileContext(const IOUringFileContext&) = delete;
    IOUringFileContext& operator=(const IOUringFileContext&) = delete;

    ~IOUringFileContext() {
        if (fd_ >= 0) close(fd_);
    }

    int getHandle() const { return fd_; }

    bool ready() const { return ready_; }

   private:
    int fd_;
    bool ready_;
};

IOUringTransport::IOUringTransport() : installed_(false) {}

IOUringTransport::~IOUringTransport() { uninstall(); }

Status IOUringTransport::install(std::string& local_segment_name,
                                 std::shared_ptr<ControlService> metadata,
                                 std::shared_ptr<Topology> local_topology,
                                 std::shared_ptr<Config> conf) {
    if (installed_) {
        return Status::InvalidArgument(
            "IO Uring transport has been installed" LOC_MARK);
    }

    CHECK_STATUS(probeCapabilities());
    metadata_ = metadata;
    local_segment_name_ = local_segment_name;
    local_topology_ = local_topology;
    conf_ = conf;
    installed_ = true;
    async_memcpy_threshold_ =
        conf_->get("transports/nvlink/async_memcpy_threshold", 1024) * 1024;
    caps.dram_to_file = true;
    if (Platform::getLoader().type() == "cuda") {
        caps.gpu_to_file = true;
    }
    return Status::OK();
}

Status IOUringTransport::probeCapabilities() {
    struct io_uring probe_ring;
    int rc = io_uring_queue_init(2, &probe_ring, 0);
    if (rc < 0) {
        LOG(INFO) << "IOUringTransport: io_uring_queue_init failed: "
                  << strerror(-rc);
        return Status::InternalError("io_uring not supported on this kernel");
    }
    io_uring_queue_exit(&probe_ring);
    return Status::OK();
}

Status IOUringTransport::uninstall() {
    if (installed_) {
        metadata_.reset();
        installed_ = false;
    }
    return Status::OK();
}

Status IOUringTransport::allocateSubBatch(SubBatchRef& batch, size_t max_size) {
    auto io_uring_batch = Slab<IOUringSubBatch>::Get().allocate();
    if (!io_uring_batch)
        return Status::InternalError("Unable to allocate IO Uring sub-batch");
    batch = io_uring_batch;
    io_uring_batch->max_size = max_size;
    io_uring_batch->task_list.reserve(max_size);
    int rc = io_uring_queue_init(max_size, &io_uring_batch->ring, 0);
    if (rc)
        return Status::InternalError(
            std::string("io_uring_queue_init failed: ") + strerror(-rc) +
            LOC_MARK);
    return Status::OK();
}

Status IOUringTransport::freeSubBatch(SubBatchRef& batch) {
    auto io_uring_batch = dynamic_cast<IOUringSubBatch*>(batch);
    if (!io_uring_batch)
        return Status::InvalidArgument("Invalid IO Uring sub-batch" LOC_MARK);
    io_uring_queue_exit(&io_uring_batch->ring);
    Slab<IOUringSubBatch>::Get().deallocate(io_uring_batch);
    batch = nullptr;
    return Status::OK();
}

std::string IOUringTransport::getIOUringFilePath(SegmentID target_id) {
    SegmentDesc* desc = nullptr;
    auto status = metadata_->segmentManager().getRemoteCached(desc, target_id);
    if (!status.ok() || desc->type != SegmentType::File) return "";
    auto& detail = std::get<FileSegmentDesc>(desc->detail);
    if (detail.buffers.empty()) return "";
    return detail.buffers[0].path;
}

IOUringFileContext* IOUringTransport::findFileContext(SegmentID target_id) {
    thread_local FileContextMap tl_file_context_map;
    if (tl_file_context_map.count(target_id))
        return tl_file_context_map[target_id].get();

    RWSpinlock::WriteGuard guard(file_context_lock_);
    if (!file_context_map_.count(target_id)) {
        std::string path = getIOUringFilePath(target_id);
        if (path.empty()) return nullptr;
        file_context_map_[target_id] =
            std::make_shared<IOUringFileContext>(path);
    }

    tl_file_context_map = file_context_map_;
    return tl_file_context_map[target_id].get();
}

Status IOUringTransport::submitTransferTasks(
    SubBatchRef batch, const std::vector<Request>& request_list) {
    auto io_uring_batch = dynamic_cast<IOUringSubBatch*>(batch);
    if (!io_uring_batch)
        return Status::InvalidArgument("Invalid IO Uring sub-batch" LOC_MARK);
    if (request_list.size() + (int)io_uring_batch->task_list.size() >
        io_uring_batch->max_size)
        return Status::TooManyRequests("Exceed batch capacity" LOC_MARK);
    for (auto& request : request_list) {
        io_uring_batch->task_list.push_back(IOUringTask{});
        auto& task =
            io_uring_batch->task_list[io_uring_batch->task_list.size() - 1];
        task.request = request;
        task.status_word = TransferStatusEnum::PENDING;

        IOUringFileContext* context = findFileContext(request.target_id);
        if (!context || !context->ready())
            return Status::InvalidArgument("Invalid remote segment" LOC_MARK);

        task.fd = context->getHandle();
        task.bytes_completed = 0;

        struct io_uring_sqe* sqe = io_uring_get_sqe(&io_uring_batch->ring);
        if (!sqe)
            return Status::InternalError("io_uring_get_sqe failed" LOC_MARK);

        const size_t kPageSize = 4096;
        if (Platform::getLoader().getMemoryType(request.source) == MTYPE_CUDA ||
            (uint64_t)request.source % kPageSize) {
            int rc = posix_memalign(&task.buffer, kPageSize, request.length);
            if (rc)
                return Status::InternalError("posix_memalign failed" LOC_MARK);

            if (request.opcode == Request::READ)
                io_uring_prep_read(sqe, context->getHandle(), task.buffer,
                                   request.length, request.target_offset);
            else if (request.opcode == Request::WRITE) {
                Platform::getLoader().copy(task.buffer, request.source,
                                           request.length);
                io_uring_prep_write(sqe, context->getHandle(), task.buffer,
                                    request.length, request.target_offset);
            }
        } else {
            if (request.opcode == Request::READ)
                io_uring_prep_read(sqe, context->getHandle(), request.source,
                                   request.length, request.target_offset);
            else if (request.opcode == Request::WRITE)
                io_uring_prep_write(sqe, context->getHandle(), request.source,
                                    request.length, request.target_offset);
        }
        sqe->user_data = (uintptr_t)&task;
    }

    int rc = io_uring_submit(&io_uring_batch->ring);
    if (rc != (int32_t)request_list.size())
        return Status::InternalError(std::string("io_uring_submit failed: ") +
                                     strerror(-rc) + LOC_MARK);

    return Status::OK();
}

Status IOUringTransport::getTransferStatus(SubBatchRef batch, int task_id,
                                            TransferStatus& status) {
    auto io_uring_batch = dynamic_cast<IOUringSubBatch*>(batch);
    if (task_id < 0 || task_id >= (int)io_uring_batch->task_list.size())
        return Status::InvalidArgument("Invalid task ID");
    auto& task = io_uring_batch->task_list[task_id];
    status = TransferStatus{task.status_word, task.transferred_bytes};
    if (task.status_word == TransferStatusEnum::PENDING) {
        struct io_uring_cqe* cqe = nullptr;
        int err = io_uring_peek_cqe(&io_uring_batch->ring, &cqe);
        if (err == -EAGAIN) return Status::OK();
        if (err || !cqe) {
            return Status::InternalError(
                std::string("io_uring_peek_cqe failed: ") + strerror(-err));
        }
        auto task = (IOUringTask*)cqe->user_data;
        if (task) {
            if (cqe->res < 0) {
                LOG(ERROR) << "io_uring I/O error: " << cqe->res
                           << " (" << strerror(-cqe->res) << ")"
                           << " op=" << (task->request.opcode == Request::WRITE
                                             ? "WRITE"
                                             : "READ")
                           << " offset=" << task->request.target_offset
                           << " len=" << task->request.length
                           << " completed_so_far=" << task->bytes_completed;
                task->status_word = TransferStatusEnum::FAILED;
                // Clean up bounce buffer on failure
                if (task->buffer) {
                    free(task->buffer);
                    task->buffer = nullptr;
                }
            } else if (cqe->res == 0 &&
                       task->bytes_completed < task->request.length) {
                // Zero-length completion with outstanding bytes is an error
                // (e.g. writing past EOF without O_APPEND)
                LOG(ERROR) << "io_uring short I/O: 0 bytes returned with "
                           << (task->request.length - task->bytes_completed)
                           << " bytes remaining";
                task->status_word = TransferStatusEnum::FAILED;
                if (task->buffer) {
                    free(task->buffer);
                    task->buffer = nullptr;
                }
            } else {
                task->bytes_completed += (size_t)cqe->res;
                size_t remaining =
                    task->request.length - task->bytes_completed;

                if (remaining > 0) {
                    // Short write/read: resubmit for the remaining bytes
                    LOG(WARNING)
                        << "io_uring short I/O: got " << cqe->res
                        << " bytes, " << remaining << " remaining"
                        << " (total " << task->bytes_completed << "/"
                        << task->request.length << ")";

                    struct io_uring_sqe* sqe =
                        io_uring_get_sqe(&io_uring_batch->ring);
                    if (!sqe) {
                        LOG(ERROR) << "io_uring_get_sqe failed during "
                                      "short-write resubmission";
                        task->status_word = TransferStatusEnum::FAILED;
                    } else {
                        uint64_t new_offset = task->request.target_offset +
                                              task->bytes_completed;
                        // Determine the I/O buffer pointer for the remainder.
                        // If using a bounce buffer, advance into it;
                        // otherwise advance the source pointer.
                        void* io_buf;
                        if (task->buffer)
                            io_buf = (char*)task->buffer +
                                     task->bytes_completed;
                        else
                            io_buf = (char*)task->request.source +
                                     task->bytes_completed;

                        if (task->request.opcode == Request::WRITE)
                            io_uring_prep_write(sqe, task->fd, io_buf,
                                                remaining, new_offset);
                        else
                            io_uring_prep_read(sqe, task->fd, io_buf,
                                               remaining, new_offset);

                        sqe->user_data = (uintptr_t)task;
                        int rc = io_uring_submit(&io_uring_batch->ring);
                        if (rc < 1) {
                            LOG(ERROR)
                                << "io_uring_submit failed during "
                                   "short-write resubmission: "
                                << strerror(-rc);
                            task->status_word = TransferStatusEnum::FAILED;
                        }
                        // else: stays PENDING, will be polled again
                    }
                } else {
                    // All bytes transferred successfully
                    if (task->buffer) {
                        if (task->request.opcode == Request::READ)
                            Platform::getLoader().copy(
                                task->request.source, task->buffer,
                                task->request.length);
                        free(task->buffer);
                        task->buffer = nullptr;
                    }
                    task->status_word = TransferStatusEnum::COMPLETED;
                    task->transferred_bytes = task->request.length;
                }
            }
        }
        io_uring_cqe_seen(&io_uring_batch->ring, cqe);
        if (task)
            status = TransferStatus{task->status_word,
                                    task->transferred_bytes};
    }
    return Status::OK();
}

Status IOUringTransport::addMemoryBuffer(BufferDesc& desc,
                                         const MemoryOptions& options) {
    return Status::OK();
}

Status IOUringTransport::removeMemoryBuffer(BufferDesc& desc) {
    return Status::OK();
}

}  // namespace tent
}  // namespace mooncake
