const c = @import("c.zig").c;

pub fn get_vk_platform_extension() [*c]const u8 {
    return c.VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME;
}
