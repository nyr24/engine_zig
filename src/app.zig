const std = @import("std");
const logger_mod = @import("logger.zig");
const Arena = std.heap.ArenaAllocator;
const StackAllocator = @import("stack_allocator.zig").StackAllocator;

pub const Application = struct {
    arena_alloc: Arena,
    stack_alloc: StackAllocator(.{}),

    const Self = @This();

    pub fn init() !Application {
        var app: Application = undefined;
        app.arena_alloc = Arena.init(std.heap.c_allocator);
        app.stack_alloc = try .init_capacity(0xffffffff);

        return app;
    }

    pub fn deinit(self: *Self) void {
        self.arena_alloc.deinit();
        self.stack_alloc.deinit();
    }
};
