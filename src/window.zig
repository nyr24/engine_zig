const std = @import("std");
const c = @import("c.zig").c;
const vk_shared = @import("vk_shared.zig");
const logger = @import("logger.zig");
const VulkanContext = vk_shared.VulkanContext;
const vk_check = vk_shared.vk_check;

pub const Window = struct {
    const Self = @This();
    handle: ?*c.GLFWwindow,
    width: u32,
    height: u32,

    pub fn init(width: u32, height: u32, name: []const u8) Self {
        if (c.glfwInit() == 0) {
            @branchHint(.cold);
            logger.log_error(.FATAL, "glfw failed to initialize", .{});
            @panic("");
        }
        if (c.glfwVulkanSupported() == 0) {
            @branchHint(.cold);
            logger.log_error(.FATAL, "vulkan is not supported for this version of glfw", .{});
            @panic("");
        }

        var window: Self = undefined;
        window.width = width;
        window.height = height;

        c.glfwWindowHint(c.GLFW_CLIENT_API, c.GLFW_NO_API);

        window.handle = c.glfwCreateWindow(@intCast(width), @intCast(height), name.ptr, null, null) orelse {
            logger.log_error(.FATAL, "window creation is failed", .{});
            @panic("");
        };

        c.glfwSetInputMode(window.handle, c.GLFW_CURSOR, c.GLFW_CURSOR_DISABLED);
        _ = c.glfwSetKeyCallback(window.handle, &key_callback);
        _ = c.glfwSetMouseButtonCallback(window.handle, &mouse_btn_callback);
        _ = c.glfwSetCursorPosCallback(window.handle, &mouse_move_callback);
        _ = c.glfwSetScrollCallback(window.handle, &mouse_wheel_callback);

        return window;
    }

    pub fn deinit(self: *Self) void {
        if (self.handle != null) {
            c.glfwDestroyWindow(self.handle);
            self.handle = null;
        }
        c.glfwTerminate();
    }
};

pub fn create_vk_surface(context: *VulkanContext, window: Window) void {
    vk_check(c.glfwCreateWindowSurface(context.instance.handle, window.handle, null, &context.surface));
}

fn key_callback(window: ?*c.GLFWwindow, key: c_int, scancode: c_int, action: c_int, mode: c_int) callconv(.c) void {
    _ = window;
    _ = scancode;
    _ = mode;

    if (action == c.GLFW_REPEAT) {
        return;
    }

    // TODO: input process
    _ = key;
    // input_process_key(key, action == GLFW_PRESS);
}

fn mouse_move_callback(window: ?*c.GLFWwindow, x: f64, y: f64) callconv(.c) void {
    _ = window;
    // TODO: input process
    _ = x;
    _ = y;
    // input_process_mouse_move(MousePos{ static_cast<f32>(x), static_cast<f32>(y) });
}

fn mouse_btn_callback(window: ?*c.GLFWwindow, btn: c_int, action: c_int, mods: c_int) callconv(.c) void {
    _ = window;
    _ = mods;

    // TODO: input process
    _ = btn;
    _ = action;
    // input_process_mouse_button(static_cast<MouseButton>(btn), action == GLFW_PRESS);
}

fn mouse_wheel_callback(window: ?*c.GLFWwindow, x_offset: f64, y_offset: f64) callconv(.c) void {
    _ = window;
    _ = x_offset;
    _ = y_offset;

    // if (std.math.complex.abs(y_offset) > std.math.float.floatEps(f64)) {
    // TODO: input process
    // input_process_mouse_wheel(yoffset > 0 ? -1 : 1);
    // }
}
