//! Dedicated OS thread that owns the [`Live2DBackend`] and runs every
//! `render_frame` call off the tokio worker pool.
//!
//! `WgpuBackend::render_frame` blocks the calling thread on
//! `wgpu::Device::poll(Maintain::Wait)` while the GPU completes a
//! texture readback. Calling that inline from an `async fn` parks
//! the tokio worker for the entire GPU sync, starving co-located
//! tasks (notably the `audio_sender` ticker on the same runtime).
//!
//! Mirrors the iceoryx2 `!Send` pattern in
//! `multiprocess_executor`: keep the resource on a single OS thread
//! for its lifetime; communicate via channels.

use super::backend_trait::{BackendError, Live2DBackend, RgbFrame};
use super::state::Pose;
use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use tokio::sync::oneshot;

struct RenderRequest {
    pose: Pose,
    response: oneshot::Sender<Result<RgbFrame, BackendError>>,
}

/// Owns the backend on a dedicated thread. Drop the worker to stop
/// the thread (the recv loop exits when the request channel closes).
pub(super) struct RenderWorker {
    request_tx: mpsc::Sender<RenderRequest>,
    // Held only so the JoinHandle isn't dropped immediately; the
    // thread is detached on drop and exits naturally when
    // `request_tx` is dropped.
    _join: JoinHandle<()>,
}

impl RenderWorker {
    pub fn spawn(mut backend: Box<dyn Live2DBackend + Send>) -> Self {
        let (tx, rx) = mpsc::channel::<RenderRequest>();
        let join = thread::Builder::new()
            .name("live2d-render".into())
            .spawn(move || {
                while let Ok(req) = rx.recv() {
                    let result = backend.render_frame(&req.pose);
                    // Receiver may have been dropped if the caller
                    // task was cancelled mid-await; that's fine.
                    let _ = req.response.send(result);
                }
            })
            .expect("failed to spawn live2d-render thread");
        Self {
            request_tx: tx,
            _join: join,
        }
    }

    /// Submit a pose to the render thread and await the resulting
    /// frame. The caller's tokio worker is freed during the GPU
    /// sync.
    pub async fn render(&self, pose: Pose) -> Result<RgbFrame, BackendError> {
        let (tx, rx) = oneshot::channel();
        self.request_tx
            .send(RenderRequest { pose, response: tx })
            .map_err(|_| BackendError::Other("live2d render worker disconnected".into()))?;
        rx.await
            .map_err(|_| BackendError::Other("live2d render worker dropped response".into()))?
    }
}
