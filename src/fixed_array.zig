const std = @import("std");
const builtin = @import("builtin");

pub fn FixedArray(comptime T: type, CAPACITY: u32) type {
    return struct {
        const Self = @This();

        comptime capacity: u32 = CAPACITY,
        len: u32 = 0,
        items: [CAPACITY]T = undefined,

        pub const empty = Self{};

        pub fn init_len(input_len: u32) Self {
            std.debug.assert(input_len <= CAPACITY);
            return Self{ .len = input_len };
        }

        pub fn init_len_to_cap() Self {
            return Self{ .len = CAPACITY };
        }

        pub fn init_from_slice(input_slice: []const T) Self {
            var new_arr = empty;
            const append_items = blk: {
                if (input_slice.len > CAPACITY) {
                    break :blk input_slice[0..CAPACITY];
                } else {
                    break :blk input_slice;
                }
            };

            new_arr.append_many(append_items);
            return new_arr;
        }

        pub fn append(self: *Self, item: T) void {
            if (self.len == self.capacity) {
                return;
            }

            self.items[self.len] = item;
            self.len += 1;
        }

        pub fn append_many(self: *Self, items: []const T) void {
            for (items) |item| {
                self.append(item);
            }
        }

        pub fn slice(self: *Self, start: u32, end: u32) []T {
            std.debug.assert(start >= 0 and end <= self.len);
            return self.items[start..end];
        }

        pub fn slice_as_const(self: *const Self, start: u32, end: u32) []const T {
            std.debug.assert(start >= 0 and end <= self.len);
            return self.items[start..end];
        }

        pub fn slice_full(self: *Self) []T {
            return self.items[0..self.len];
        }

        pub fn slice_full_as_const(self: *const Self) []const T {
            return self.items[0..self.len];
        }

        pub fn clear(self: *Self) void {
            self.len = 0;
        }

        pub fn resize(self: *Self, input_len: u32) void {
            std.debug.assert(input_len <= self.capacity);
            self.len = input_len;
        }

        pub fn resize_to_cap(self: *Self) void {
            self.len = self.capacity;
        }

        pub fn data(self: *Self) *T {
            return @ptrCast(&self.items);
        }

        pub fn data_as_const(self: *const Self) *const T {
            return @ptrCast(&self.items);
        }

        pub fn c_ptr(self: *Self) [*c]T {
            return @ptrCast(&self.items);
        }

        pub fn c_ptr_as_const(self: *const Self) [*c]const T {
            return @ptrCast(&self.items);
        }
    };
}
