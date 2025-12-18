const std = @import("std");
const util = @import("util.zig");
const log = @import("logger.zig").log_debug;

// TODO: test between Self and *Self
pub fn Vec(comptime LEN: u32, comptime T: type) type {
    return struct {
        v: @Vector(LEN, T) = undefined,

        const Self = @This();
        const len = LEN;
        const zero_vec: Self = .{ .v = @splat(0) };
        const unit_vec: Self = .{ .v = @splat(1) };

        pub fn init_from_slice(data: *const [LEN]f32) Self {
            var res: Self = .{};
            for (data, 0..LEN) |val, i| {
                res.v[i] = val;
            }
            return res;
        }

        pub fn init_from_vec(input_v: Vec(LEN, f32)) Self {
            return .{ .v = input_v };
        }

        pub fn init_dupl(val: f32) Self {
            return Self{ .v = @splat(val) };
        }

        pub fn init_zero() Self {
            return zero_vec;
        }

        pub fn init_one() Self {
            return unit_vec;
        }

        pub fn set(self: *Self, input: *const [LEN]f32) void {
            for (0..LEN) |i| {
                self.v[i] = input[i];
            }
        }

        pub inline fn x(self: Self) f32 {
            return self.v[0];
        }

        pub inline fn y(self: Self) f32 {
            return self.v[1];
        }

        pub inline fn z(self: Self) f32 {
            return self.v[2];
        }

        pub fn cross(lhs: Self, rhs: Self) Self {
            if (LEN == 3) {
                const lx, const ly, const lz = lhs.v;
                const rx, const ry, const rz = rhs.v;
                return .{ ly * rz - lz * ry, lx * rz - lz * rx, lx * ry - ly * rx };
            }
            unreachable;
        }

        pub fn dot(lhs: Self, rhs: Self) f32 {
            return @reduce(.Add, lhs.v * rhs.v);
        }

        pub fn normalized(self: Self) Self {
            var res: Self = self;
            res.negate_inplace();
            return res;
        }

        pub fn normalize_inplace(self: *Self) void {
            const mag = self.magnitude();
            self.v *= 1 / mag;
        }

        pub inline fn magnitude(self: Self) f32 {
            return @sqrt(self.magnitude_squared());
        }

        pub inline fn magnitude_squared(self: Self) f32 {
            return self.dot(self);
        }

        pub fn distance(lhs: Self, rhs: Self) Self {
            return rhs.v - lhs.v;
        }

        pub fn negated(self: Self) Self {
            var res: Self = self;
            res.negate_inplace();
            return res;
        }

        pub inline fn negate_inplace(self: Self) Self {
            self.v *= -1.0;
        }

        pub fn to_vec4(input: Vec3) Vec4 {
            var res: Vec4 = .{};
            for (0..Vec3.len) |i| {
                res.v[i] = input.v[i];
            }
            res.v[3] = 1;
            return res;
        }

        pub fn to_vec3(input: Vec4) Vec3 {
            var res: Vec3 = .{};
            res.v = .{ input.v[0], input.v[1], input.v[2] };
            return res;
        }

        pub fn cmp(lhs: Self, rhs: Self) bool {
            return lhs.v == rhs.v;
        }

        pub fn print(self: Self) void {
            log(.INFO, "{}", .{self.v});
        }
    };
}

pub const Vec3 = Vec(3, f32);
pub const Vec4 = Vec(4, f32);

// Mat4x4 -----------------------------------------------------------------------------------

pub const Axis = enum { X, Y, Z };

// Matrix-vector multiplication scheme will be: Mat4x4 * Vec4
// NOTE: coordinate systems in Vulkan https://www.kdab.com/projection-matrices-with-vulkan-part-1/

// TODO: translate, rotate, scale on existing matrix, transpose
// TODO: test different representations ([16]f32, [4]@Vector(4, f32), @Vector(16, f32))
pub const Matrix4x4 = struct {
    const ROWS = 4;
    const COLS = 4;
    const ITEMS = ROWS * COLS;
    const Self = @This();

    m: [ROWS]Vec4 = std.mem.zeroes([ROWS]Vec4),

    pub fn init_from_slice(input_slice: *const [ITEMS]f32) Self {
        var res = Self{};
        for (0..ROWS) |y| {
            for (0..COLS) |x| {
                res.m[y].v[x] = input_slice[y * COLS + x];
            }
        }
        return res;
    }

    pub fn init_from_rows(input_rows: *const [ROWS]Vec4) Self {
        var res = Self{};
        for (0..ROWS) |y| {
            res.m[y] = .init_from_vec(input_rows[y]);
        }
        return res;
    }

    pub fn init_from_row_slices(input_rows: *const [ROWS][COLS]f32) Self {
        var res = Self{};
        for (0..ROWS) |y| {
            res.m[y] = .init_from_slice(&input_rows[y]);
        }
        return res;
    }

    pub fn init_identity() Self {
        var res = Self{};
        res.set_at(1.0, 0, 0);
        res.set_at(1.0, 1, 1);
        res.set_at(1.0, 2, 2);
        res.set_at(1.0, 3, 3);
        return res;
    }

    pub fn init_translation(translation_vec: Vec3) Self {
        var res: Self = .init_identity();
        res.set_col_vec(translation_vec.to_vec4(), 3);
        return res;
    }

    pub fn translate(self: *Self, translation_vec: Vec3) void {
        self.increment_col(translation_vec.to_vec4(), 3);
        return self;
    }

    pub fn init_scaling(scaling_vec: Vec3) Self {
        var res: Self = .{};
        for (0..3) |i| {
            res.set_at(scaling_vec[i], i, i);
        }
        return res;
    }

    pub fn scale(self: *Self, scaling_vec: Vec3) void {
        for (0..3) |i| {
            self.increment_at(scaling_vec[i], i, i);
        }
    }

    pub fn init_rotation(rotate_deg: f32, axis: Axis) Self {
        var res: Self = .{};
        const sin = @sin(std.math.rad_per_deg * rotate_deg);
        const cos = @cos(std.math.rad_per_deg * rotate_deg);

        switch (axis) {
            .X => {
                res.set_at(cos, 1, 1);
                res.set_at(-sin, 1, 2);
                res.set_at(sin, 2, 1);
                res.set_at(cos, 2, 2);
            },
            .Y => {
                res.set_at(cos, 0, 0);
                res.set_at(sin, 0, 2);
                res.set_at(-sin, 2, 0);
                res.set_at(cos, 2, 2);
            },
            .Z => {
                res.set_at(cos, 0, 0);
                res.set_at(-sin, 0, 1);
                res.set_at(sin, 1, 0);
                res.set_at(cos, 1, 1);
            },
        }

        return res;
    }

    pub fn rotate(self: *Self, rotate_deg: f32, axis: Axis) void {
        const sin = @sin(std.math.rad_per_deg * rotate_deg);
        const cos = @cos(std.math.rad_per_deg * rotate_deg);

        switch (axis) {
            .X => {
                self.increment_at(cos, 1, 1);
                self.increment_at(-sin, 1, 2);
                self.increment_at(sin, 2, 1);
                self.increment_at(cos, 2, 2);
            },
            .Y => {
                self.increment_at(cos, 0, 0);
                self.increment_at(sin, 0, 2);
                self.increment_at(-sin, 2, 0);
                self.increment_at(cos, 2, 2);
            },
            .Z => {
                self.increment_at(cos, 0, 0);
                self.increment_at(-sin, 0, 1);
                self.increment_at(sin, 1, 0);
                self.increment_at(cos, 1, 1);
            },
        }
    }

    pub fn mul(lhs: *Self, rhs: *Self) Self {
        var res = Self{};

        for (0..ROWS) |y| {
            const lv = lhs.m[y];
            for (0..COLS) |x| {
                const rv: Vec4 = rhs.get_col(x);
                const d = lv.dot(rv);
                res.set_at(d, y, x);
            }
        }

        return res;
    }

    // NOTE: eliminates extra matrix in between of calculation, maybe performance benefit
    pub fn mul_many(first: Self, inputs: []const Self) Self {
        var res = Self{};

        for (0..ROWS) |y| {
            var row: Vec4 = first.get_row(y);
            var col: Vec4 = undefined;
            var between_res: Vec4 = undefined;

            for (inputs) |input_mat| {
                for (0..COLS) |x| {
                    col = input_mat.get_col(x);
                    between_res[x] = row.dot(col);
                }
                row = between_res;
            }

            res.set_row(row, y);
        }

        return res;
    }

    pub fn transpose(self: *Self) void {
        std.mem.swap(f32, self.get_at_ptr(0, 1), self.get_at_ptr(1, 0));
        std.mem.swap(f32, self.get_at_ptr(0, 2), self.get_at_ptr(2, 0));
        std.mem.swap(f32, self.get_at_ptr(0, 3), self.get_at_ptr(3, 0));
        std.mem.swap(f32, self.get_at_ptr(1, 2), self.get_at_ptr(2, 1));
        std.mem.swap(f32, self.get_at_ptr(1, 3), self.get_at_ptr(3, 1));
        std.mem.swap(f32, self.get_at_ptr(2, 3), self.get_at_ptr(3, 2));
    }

    pub fn transposed(self: Self) Self {
        var res: Self = self;
        std.mem.swap(f32, res.get_at_ptr(0, 1), res.get_at_ptr(1, 0));
        std.mem.swap(f32, res.get_at_ptr(0, 2), res.get_at_ptr(2, 0));
        std.mem.swap(f32, res.get_at_ptr(0, 3), res.get_at_ptr(3, 0));
        std.mem.swap(f32, res.get_at_ptr(1, 2), res.get_at_ptr(2, 1));
        std.mem.swap(f32, res.get_at_ptr(1, 3), res.get_at_ptr(3, 1));
        std.mem.swap(f32, res.get_at_ptr(2, 3), res.get_at_ptr(3, 2));
        return res;
    }

    pub fn init_look_at(pos: Vec3, target: Vec3, world_up: Vec3) Self {
        const dir = target.distance(pos);
        const right = world_up.cross(dir);
        const up = dir.cross(right);

        var res: Self = .init_identity();
        res.set_row(right, 0);
        res.set_row(up, 1);
        res.set_row(dir, 2);

        res.translate(pos.negated());

        return res;
    }

    // NOTE: article reference https://www.kdab.com/projection-matrices-with-vulkan-part-2/
    pub fn projection(t: f32, b: f32, l: f32, r: f32, n: f32, f: f32) Self {
        var res: Self = .{};
        res.set_at((2.0 * n) / (r - l), 0, 0);
        res.set_at(-((r + l) / (r - l)), 0, 2);
        res.set_at((2.0 * n) / (b - t), 1, 1);
        res.set_at(-((b + t) / (b - t)), 1, 2);
        res.set_at(f / (f - n), 2, 2);
        res.set_at(-((n * f) / (f - n)), 2, 3);
        res.set_at(1.0, 3, 2);
        return res;
    }

    pub fn projection_fov(fov: f32, aspect: f32, n: f32, f: f32) Self {
        var res: Self = .{};
        const fov_half = fov / 2;
        const tan = @tan(fov_half);
        const b = tan * n;
        const r = b * aspect;

        res.set_at(n / r, 0, 0);
        res.set_at(n / b, 1, 1);
        res.set_at(f / (f - n), 2, 2);
        res.set_at(-((n * f) / (f - n)), 2, 3);
        res.set_at(1.0, 3, 2);

        return res;
    }

    pub inline fn set_row(self: *Self, input_row: Vec4, index: usize) void {
        self.m[index] = input_row;
    }

    pub inline fn increment_row(self: *Self, input_row: Vec4, index: usize) void {
        self.m[index].v += input_row.v;
    }

    pub inline fn set_col(self: *Self, input_col: Vec4, index: usize) void {
        for (0..ROWS) |i| {
            self.set_at(input_col[i], i, index);
        }
    }

    pub inline fn increment_col(self: *Self, input_col: Vec4, index: usize) void {
        for (0..ROWS) |i| {
            self.increment_at(input_col[i], i, index);
        }
    }

    pub inline fn get_row(self: *const Self, index: usize) Vec4 {
        return self.m[index];
    }

    pub fn get_col(self: *Self, index: usize) Vec4 {
        var col: Vec4 = .{};
        for (0..ROWS) |i| {
            col.v[i] = self.m[i].v[index];
        }
        return col;
    }

    pub inline fn set_at(self: *Self, val: f32, row: usize, col: usize) void {
        self.m[row].v[col] = val;
    }

    pub inline fn increment_at(self: *Self, val: f32, row: usize, col: usize) void {
        self.m[row].v[col] += val;
    }

    pub inline fn get_at(self: *const Self, row: usize, col: usize) f32 {
        return self.m[row].v[col];
    }

    pub inline fn get_at_ptr(self: *const Self, row: usize, col: usize) f32 {
        return self.m[row].v[col];
    }

    pub fn cmp(lhs: Self, rhs: Self) bool {
        for (0..ROWS) |y| {
            const l_row = lhs.get_row(y);
            const r_row = rhs.get_row(y);
            if (!l_row.cmp(r_row)) {
                return false;
            }
        }
        return true;
    }

    pub fn print(self: Self) void {
        for (0..ROWS) |y| {
            self.get_row(y).print();
        }
    }
};
