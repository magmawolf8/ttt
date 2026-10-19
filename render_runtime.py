"""Keep hybrid-laptop OpenGL selection consistent with standalone episodes."""
from contextlib import contextmanager

from mujoco.gl_context import GLContext
from OpenGL import GL


def current_render_backend():
    """Query the current graphics context; this does not change policy devices."""
    def value(name):
        result=GL.glGetString(name)
        return None if result is None else result.decode()
    return {"vendor":value(GL.GL_VENDOR),"renderer":value(GL.GL_RENDERER),
            "version":value(GL.GL_VERSION)}


@contextmanager
def graphics_before_cuda():
    """Create and retain a tiny GL context before importing Torch.

    On this hybrid laptop, even importing Torch before the first GL context
    selects a different rendering adapter. A retained bootstrap context keeps batch
    camera pixels on the same path as run_policy.py's graphics-first startup.
    """
    context=GLContext(1,1)
    try:
        context.make_current()
        yield current_render_backend()
    finally:
        context.make_current()
        context.free()
