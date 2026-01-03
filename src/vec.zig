const std = @import("std");
const math = std.math;
const logger = @import("logger.zig");

// NOTE: All vectors, 2, 3, 4 merged into 1 type, because it's easy to share operations between them
pub fn Vec(comptime LEN: u32, comptime T: type) type {
    if (@typeInfo(T) != .float and @typeInfo(T) != .int) {
        @compileError("Vectors not implemented for " ++ @typeName(T));
    }

    if (LEN < 2 or LEN > 4) {
        @compileError("Dimensions must be 2, 3 or 4!");
    }

    // NOTE: test extern
    return extern struct {
        const Self = @This();
        const VecT = @Vector(LEN, T);
        pub const len = LEN;
        pub const zero_vec: Self = .{ .data = @splat(0) };
        pub const unit_vec: Self = .{ .data = @splat(1) };

        data: VecT = undefined,

        pub const DimensionImpl = switch (LEN) {
            2 => struct {
                pub fn init(x_: f32, y_: f32) Self {
                    return .{ .data = .{ x_, y_ } };
                }

                pub fn to_vec3(self: Vec2) Vec3 {
                    return Vec3.init(self.x(), self.y(), 1.0);
                }

                pub fn from_vec3(v3: Vec3) Vec2 {
                    return Vec2.init(v3.x(), v3.y());
                }

                pub fn to_vec4(self: Vec2) Vec4 {
                    return Vec4.init(self.x(), self.y(), 1.0, 1.0);
                }

                pub fn from_vec4(v4: Vec4) Vec2 {
                    return Vec2.init(v4.x(), v4.y());
                }

                pub fn swizzle_yx(self: Self) Vec2 {
                    return Vec2.init(self.y(), self.x());
                }

                pub const right: Self = .{ .data = .{ 1.0, 0.0 } };
                pub const left: Self = .{ .data = .{ -1.0, 0.0 } };
                pub const top: Self = .{ .data = .{ 0.0, -1.0 } };
                pub const bottom: Self = .{ .data = .{ 0.0, 1.0 } };
            },
            3 => struct {
                pub fn init(x_: f32, y_: f32, z_: f32) Self {
                    return .{ .data = .{ x_, y_, z_ } };
                }

                pub fn to_vec2(self: Vec3) Vec2 {
                    return Vec2.init(self.x(), self.y());
                }

                pub fn to_vec4(self: Vec3) Vec4 {
                    return Vec4.init(self.x(), self.y(), self.z(), 1.0);
                }

                pub fn as_vec2(self: *Vec3) *Vec2 {
                    return @as(*Vec2, @ptrCast(@alignCast(self)));
                }

                pub fn as_vec2_as_const(self: *const Vec3) *const Vec2 {
                    return @as(*const Vec2, @ptrCast(@alignCast(self)));
                }

                pub fn from_vec2(v2: Vec2) Vec3 {
                    return Vec3.init(v2.x(), v2.y(), 1.0);
                }

                pub fn from_vec4(v4: Vec4) Vec3 {
                    return Vec3.init(v4.x(), v4.y(), v4.z());
                }

                pub inline fn z(self: Vec3) f32 {
                    return self.data[2];
                }

                pub fn cross(lhs: Vec3, rhs: Vec3) Vec3 {
                    return .{ .data = .{
                        lhs.y() * rhs.z() - lhs.z() * rhs.y(),
                        lhs.x() * rhs.z() - lhs.z() * rhs.x(),
                        lhs.x() * rhs.y() - lhs.y() * rhs.x(),
                    } };
                }

                // 2d swizzles
                pub fn swizzle_xy(self: Self) Vec2 {
                    return Vec2.init(self.x(), self.y());
                }

                pub fn swizzle_xz(self: Self) Vec2 {
                    return Vec2.init(self.x(), self.z());
                }

                pub fn swizzle_yx(self: Self) Vec2 {
                    return Vec2.init(self.y(), self.x());
                }

                pub fn swizzle_yz(self: Self) Vec2 {
                    return Vec2.init(self.y(), self.z());
                }

                pub fn swizzle_zx(self: Self) Vec2 {
                    return Vec2.init(self.z(), self.x());
                }

                pub fn swizzle_zy(self: Self) Vec2 {
                    return Vec2.init(self.z(), self.y());
                }

                // 3d swizzles
                pub fn swizzle_xzy(self: Self) Vec3 {
                    return Vec3.init(self.x(), self.z(), self.y());
                }

                pub fn swizzle_yzx(self: Self) Vec3 {
                    return Vec3.init(self.y(), self.z(), self.x());
                }

                pub fn swizzle_yxz(self: Self) Vec3 {
                    return Vec3.init(self.y(), self.x(), self.z());
                }

                pub fn swizzle_zxy(self: Self) Vec3 {
                    return Vec3.init(self.z(), self.x(), self.y());
                }

                pub fn swizzle_zyx(self: Self) Vec3 {
                    return Vec3.init(self.z(), self.y(), self.x());
                }

                pub const right: Self = .{ .data = .{ 1.0, 0.0, 0.0 } };
                pub const left: Self = .{ .data = .{ -1.0, 0.0, 0.0 } };
                pub const top: Self = .{ .data = .{ 0.0, -1.0, 0.0 } };
                pub const bottom: Self = .{ .data = .{ 0.0, 1.0, 0.0 } };
                pub const forward: Self = .{ .data = .{ 0.0, 0.0, 1.0 } };
                pub const backward: Self = .{ .data = .{ 0.0, 0.0, -1.0 } };
            },
            4 => struct {
                pub fn init(x_: f32, y_: f32, z_: f32, w_: f32) Self {
                    return .{ .data = .{ x_, y_, z_, w_ } };
                }

                pub fn to_vec2(self: Vec4) Vec2 {
                    return Vec2.init(self.x(), self.y());
                }

                pub fn to_vec3(self: Vec4) Vec3 {
                    return Vec3.init(self.x(), self.y(), self.z());
                }

                pub fn from_vec2(v2: Vec2) Vec4 {
                    return Vec4.init(v2.x(), v2.y(), 1.0, 1.0);
                }

                pub fn from_vec3(v3: Vec3) Vec4 {
                    return Vec4.init(v3.x(), v3.y(), v3.z(), 1.0);
                }

                pub fn as_vec2(self: *Vec4) *Vec2 {
                    return @as(*Vec2, @ptrCast(@alignCast(self)));
                }

                pub fn as_vec2_as_const(self: *const Vec4) *const Vec2 {
                    return @as(*const Vec2, @ptrCast(@alignCast(self)));
                }

                pub fn as_vec3(self: *Vec4) *Vec3 {
                    return @as(*Vec3, @ptrCast(@alignCast(self)));
                }

                pub fn as_vec3_as_const(self: *const Vec4) *const Vec3 {
                    return @as(*const Vec3, @ptrCast(@alignCast(self)));
                }

                pub inline fn z(self: Vec4) f32 {
                    return self.data[2];
                }

                pub inline fn w(self: Vec4) f32 {
                    return self.data[3];
                }

                pub fn cross(lhs: Vec4, rhs: Vec4) Vec4 {
                    return .{ .data = .{
                        lhs.y() * rhs.z() - lhs.z() * rhs.y(),
                        lhs.x() * rhs.z() - lhs.z() * rhs.x(),
                        lhs.x() * rhs.y() - lhs.y() * rhs.x(),
                        1.0,
                    } };
                }

                // 2d swizzles
                pub fn swizzle_xx(self: Self) Vec2 {
                    return Vec2.init(self.x(), self.x());
                }

                pub fn swizzle_xy(self: Self) Vec2 {
                    return Vec2.init(self.x(), self.y());
                }

                pub fn swizzle_xz(self: Self) Vec2 {
                    return Vec2.init(self.x(), self.z());
                }

                pub fn swizzle_xw(self: Self) Vec2 {
                    return Vec2.init(self.x(), self.w());
                }

                pub fn swizzle_yx(self: Self) Vec2 {
                    return Vec2.init(self.y(), self.x());
                }

                pub fn swizzle_yy(self: Self) Vec2 {
                    return Vec2.init(self.y(), self.y());
                }

                pub fn swizzle_yz(self: Self) Vec2 {
                    return Vec2.init(self.y(), self.z());
                }

                pub fn swizzle_yw(self: Self) Vec2 {
                    return Vec2.init(self.y(), self.w());
                }

                pub fn swizzle_zx(self: Self) Vec2 {
                    return Vec2.init(self.z(), self.x());
                }

                pub fn swizzle_zy(self: Self) Vec2 {
                    return Vec2.init(self.z(), self.y());
                }

                pub fn swizzle_zz(self: Self) Vec2 {
                    return Vec2.init(self.z(), self.z());
                }

                pub fn swizzle_zw(self: Self) Vec2 {
                    return Vec2.init(self.z(), self.w());
                }

                pub fn swizzle_wx(self: Self) Vec2 {
                    return Vec2.init(self.w(), self.x());
                }

                pub fn swizzle_wy(self: Self) Vec2 {
                    return Vec2.init(self.w(), self.y());
                }

                pub fn swizzle_wz(self: Self) Vec2 {
                    return Vec2.init(self.w(), self.z());
                }

                pub fn swizzle_ww(self: Self) Vec2 {
                    return Vec2.init(self.w(), self.w());
                }

                // 3D swizzles
                pub fn swizzle_xyz(self: Self) Vec3 {
                    return Vec3.init(self.x(), self.y(), self.z());
                }

                pub fn swizzle_xzy(self: Self) Vec3 {
                    return Vec3.init(self.x(), self.z(), self.y());
                }

                pub fn swizzle_xyw(self: Self) Vec3 {
                    return Vec3.init(self.x(), self.y(), self.w());
                }

                pub fn swizzle_xwy(self: Self) Vec3 {
                    return Vec3.init(self.x(), self.w(), self.y());
                }

                pub fn swizzle_xzw(self: Self) Vec3 {
                    return Vec3.init(self.x(), self.z(), self.w());
                }

                pub fn swizzle_xwz(self: Self) Vec3 {
                    return Vec3.init(self.x(), self.w(), self.z());
                }

                pub fn swizzle_yxz(self: Self) Vec3 {
                    return Vec3.init(self.y(), self.x(), self.z());
                }

                pub fn swizzle_yzx(self: Self) Vec3 {
                    return Vec3.init(self.y(), self.z(), self.x());
                }

                pub fn swizzle_yxw(self: Self) Vec3 {
                    return Vec3.init(self.y(), self.x(), self.w());
                }

                pub fn swizzle_ywx(self: Self) Vec3 {
                    return Vec3.init(self.y(), self.w(), self.x());
                }

                pub fn swizzle_yzw(self: Self) Vec3 {
                    return Vec3.init(self.y(), self.z(), self.w());
                }

                pub fn swizzle_ywz(self: Self) Vec3 {
                    return Vec3.init(self.y(), self.w(), self.z());
                }

                pub fn swizzle_zxy(self: Self) Vec3 {
                    return Vec3.init(self.z(), self.x(), self.y());
                }

                pub fn swizzle_zyx(self: Self) Vec3 {
                    return Vec3.init(self.z(), self.y(), self.x());
                }

                pub fn swizzle_zxw(self: Self) Vec3 {
                    return Vec3.init(self.z(), self.x(), self.w());
                }

                pub fn swizzle_zwx(self: Self) Vec3 {
                    return Vec3.init(self.z(), self.w(), self.x());
                }

                pub fn swizzle_zwy(self: Self) Vec3 {
                    return Vec3.init(self.z(), self.w(), self.y());
                }

                pub fn swizzle_wzy(self: Self) Vec3 {
                    return Vec3.init(self.w(), self.z(), self.y());
                }

                pub fn swizzle_wxy(self: Self) Vec3 {
                    return Vec3.init(self.w(), self.x(), self.y());
                }

                pub fn swizzle_wyx(self: Self) Vec3 {
                    return Vec3.init(self.w(), self.y(), self.x());
                }

                pub fn swizzle_wxz(self: Self) Vec3 {
                    return Vec3.init(self.w(), self.x(), self.z());
                }

                pub fn swizzle_wzx(self: Self) Vec3 {
                    return Vec3.init(self.w(), self.z(), self.x());
                }

                // 4D swizzles
                pub fn swizzle_xyzw(self: Self) Vec4 {
                    return Vec4.init(self.x(), self.y(), self.z(), self.w());
                }

                pub fn swizzle_xzyw(self: Self) Vec4 {
                    return Vec4.init(self.x(), self.z(), self.y(), self.w());
                }

                pub fn swizzle_xwyz(self: Self) Vec4 {
                    return Vec4.init(self.x(), self.w(), self.y(), self.z());
                }

                pub fn swizzle_xwzy(self: Self) Vec4 {
                    return Vec4.init(self.x(), self.w(), self.z(), self.y());
                }

                pub fn swizzle_xzwy(self: Self) Vec4 {
                    return Vec4.init(self.x(), self.z(), self.w(), self.y());
                }

                pub fn swizzle_yxzw(self: Self) Vec4 {
                    return Vec4.init(self.y(), self.x(), self.z(), self.w());
                }

                pub fn swizzle_yzxw(self: Self) Vec4 {
                    return Vec4.init(self.y(), self.z(), self.x(), self.w());
                }

                pub fn swizzle_ywzx(self: Self) Vec4 {
                    return Vec4.init(self.y(), self.w(), self.z(), self.x());
                }

                pub fn swizzle_yzwx(self: Self) Vec4 {
                    return Vec4.init(self.y(), self.z(), self.w(), self.x());
                }

                pub fn swizzle_ywxz(self: Self) Vec4 {
                    return Vec4.init(self.y(), self.w(), self.x(), self.z());
                }

                pub fn swizzle_zxyw(self: Self) Vec4 {
                    return Vec4.init(self.z(), self.x(), self.y(), self.w());
                }

                pub fn swizzle_zyxw(self: Self) Vec4 {
                    return Vec4.init(self.z(), self.y(), self.x(), self.w());
                }

                pub fn swizzle_zwxy(self: Self) Vec4 {
                    return Vec4.init(self.z(), self.w(), self.x(), self.y());
                }

                pub fn swizzle_zwyx(self: Self) Vec4 {
                    return Vec4.init(self.z(), self.w(), self.y(), self.x());
                }

                pub fn swizzle_zywx(self: Self) Vec4 {
                    return Vec4.init(self.z(), self.y(), self.w(), self.x());
                }

                pub fn swizzle_zxwy(self: Self) Vec4 {
                    return Vec4.init(self.z(), self.x(), self.w(), self.y());
                }

                pub fn swizzle_wxyz(self: Self) Vec4 {
                    return Vec4.init(self.w(), self.x(), self.y(), self.z());
                }

                pub fn swizzle_wzyx(self: Self) Vec4 {
                    return Vec4.init(self.w(), self.z(), self.y(), self.x());
                }

                pub fn swizzle_wyxz(self: Self) Vec4 {
                    return Vec4.init(self.w(), self.y(), self.x(), self.z());
                }

                pub fn swizzle_wxzy(self: Self) Vec4 {
                    return Vec4.init(self.w(), self.x(), self.z(), self.y());
                }

                pub fn swizzle_wyzx(self: Self) Vec4 {
                    return Vec4.init(self.w(), self.y(), self.z(), self.x());
                }

                pub fn swizzle_wzxy(self: Self) Vec4 {
                    return Vec4.init(self.w(), self.z(), self.x(), self.y());
                }

                pub const right: Self = .{ .data = .{ 1.0, 0.0, 0.0, 1.0 } };
                pub const left: Self = .{ .data = .{ -1.0, 0.0, 0.0, 1.0 } };
                pub const top: Self = .{ .data = .{ 0.0, -1.0, 0.0, 1.0 } };
                pub const bottom: Self = .{ .data = .{ 0.0, 1.0, 0.0, 1.0 } };
                pub const forward: Self = .{ .data = .{ 0.0, 0.0, 1.0, 1.0 } };
                pub const backward: Self = .{ .data = .{ 0.0, 0.0, -1.0, 1.0 } };
            },
            else => unreachable,
        };

        pub fn init_from_slice(slice: *const [LEN]f32) Self {
            var res: Self = .{};
            for (slice, 0..LEN) |val, i| {
                res.data[i] = val;
            }
            return res;
        }

        pub fn init_from_vec(input_v: VecT) Self {
            return .{ .data = input_v };
        }

        pub fn init_dupl(val: f32) Self {
            return Self{ .data = @splat(val) };
        }

        pub fn init_zero() Self {
            return zero_vec;
        }

        pub fn init_one() Self {
            return unit_vec;
        }

        pub fn set_from_slice(self: *Self, input: *const [LEN]f32) void {
            for (0..LEN) |i| {
                self.data[i] = input[i];
            }
        }

        pub fn set_from_vec(self: *Self, comptime INPUT_LEN: u32, input_vec: Vec(INPUT_LEN, f32)) void {
            const min_len = comptime @min(LEN, INPUT_LEN);
            for (0..min_len) |i| {
                self.data[i] = input_vec.data[i];
            }
        }

        pub inline fn x(self: Self) f32 {
            return self.data[0];
        }

        pub inline fn y(self: Self) f32 {
            return self.data[1];
        }

        pub inline fn dot(lhs: Self, rhs: Self) f32 {
            return @reduce(.Add, lhs.data * rhs.data);
        }

        pub fn get_angle(lhs: Self, rhs: Self) f32 {
            const dot_product = lhs.dot(rhs);
            return math.radiansToDegrees(math.acos(dot_product));
        }

        pub fn mul(lhs: Self, rhs: Self) Self {
            var res = lhs;
            res.mul_inplace(rhs);
            return res;
        }

        pub inline fn mul_inplace(lhs: *Self, rhs: Self) void {
            lhs.data *= rhs.data;
        }

        pub fn mul_scalar(self: Self, scalar: f32) Self {
            var res = self;
            res.mul_scalar_inplace(scalar);
            return res;
        }

        pub inline fn mul_scalar_inplace(self: *Self, scalar: f32) void {
            self.data *= @as(VecT, @splat(scalar));
        }

        pub fn add(lhs: Self, rhs: Self) Self {
            var res = lhs;
            res.add_inplace(rhs);
            return res;
        }

        pub inline fn add_inplace(lhs: *Self, rhs: Self) void {
            lhs.data += rhs.data;
        }

        pub fn sub(lhs: Self, rhs: Self) Self {
            var res = lhs;
            res.sub_inplace(rhs);
            return res;
        }

        pub fn sub_inplace(lhs: *Self, rhs: Self) void {
            lhs.data -= rhs.data;
        }

        pub fn div(lhs: Self, rhs: Self) Self {
            var res = lhs;
            res.div_inplace(rhs);
            return res;
        }

        pub fn div_inplace(lhs: *Self, rhs: Self) void {
            lhs.data /= rhs.data;
        }

        pub fn reciprocal(self: Self) Self {
            var res: Self = self;
            res.reciprocal_inplace();
            return res;
        }

        pub fn reciprocal_inplace(self: *Self) void {
            self.data = @as(VecT, @splat(1)) / self.data;
        }

        pub fn normalized(self: Self) Self {
            var res: Self = self;
            res.normalize_inplace();
            return res;
        }

        pub fn normalize_inplace(self: *Self) void {
            const mag_inv_v: VecT = @splat(1 / self.magnitude());
            self.data *= mag_inv_v;
        }

        pub inline fn magnitude(self: Self) f32 {
            return @sqrt(self.magnitude_squared());
        }

        pub inline fn magnitude_squared(self: Self) f32 {
            return self.dot(self);
        }

        pub fn distance(lhs: Self, rhs: Self) Self {
            const diff_v = rhs.sub(lhs);
            return diff_v.magnitude();
        }

        pub fn distance_squared(lhs: Self, rhs: Self) Self {
            const diff_v = rhs.sub(lhs);
            return diff_v.magnitude_squared();
        }

        pub fn negated(self: Self) Self {
            var res: Self = self;
            res.negate_inplace();
            return res;
        }

        pub inline fn negate_inplace(self: *Self) void {
            self.data *= @as(VecT, @splat(-1.0));
        }

        pub fn lerp(lhs: Self, rhs: Self, t: f32) Self {
            const from = lhs.data;
            const to = rhs.data;
            const lerp_v: VecT = @splat(t);
            const one: VecT = @splat(1);
            const result = (one - lerp_v) * from + lerp_v * to;
            return .{ .data = result };
        }

        pub fn eql(lhs: Self, rhs: Self) bool {
            return @reduce(.And, lhs.data == rhs.data);
        }

        pub fn print(self: Self) void {
            logger.log_debug(.INFO, "{d:.2}", .{self.data});
        }

        pub const init = DimensionImpl.init;
        pub const to_vec2 = DimensionImpl.to_vec2;
        pub const to_vec3 = DimensionImpl.to_vec3;
        pub const to_vec4 = DimensionImpl.to_vec4;
        pub const as_vec2 = DimensionImpl.as_vec2;
        pub const as_vec3 = DimensionImpl.as_vec3;
        pub const from_vec2 = DimensionImpl.from_vec2;
        pub const from_vec3 = DimensionImpl.from_vec3;
        pub const from_vec4 = DimensionImpl.from_vec4;
        pub const z = DimensionImpl.z;
        pub const w = DimensionImpl.w;
        pub const cross = DimensionImpl.cross;
        pub const right: Self = DimensionImpl.right;
        pub const left: Self = DimensionImpl.left;
        pub const top: Self = DimensionImpl.top;
        pub const bottom: Self = DimensionImpl.bottom;
        pub const forward: Self = DimensionImpl.forward;
        pub const backward: Self = DimensionImpl.backward;
        // 2d swizzles
        pub const swizzle_xx = DimensionImpl.swizzle_xx;
        pub const swizzle_xy = DimensionImpl.swizzle_xy;
        pub const swizzle_xz = DimensionImpl.swizzle_xz;
        pub const swizzle_xw = DimensionImpl.swizzle_xw;
        pub const swizzle_yx = DimensionImpl.swizzle_yx;
        pub const swizzle_yy = DimensionImpl.swizzle_yy;
        pub const swizzle_yz = DimensionImpl.swizzle_yz;
        pub const swizzle_yw = DimensionImpl.swizzle_yw;
        pub const swizzle_zx = DimensionImpl.swizzle_zx;
        pub const swizzle_zy = DimensionImpl.swizzle_zy;
        pub const swizzle_zz = DimensionImpl.swizzle_zz;
        pub const swizzle_zw = DimensionImpl.swizzle_zw;
        pub const swizzle_wx = DimensionImpl.swizzle_wx;
        pub const swizzle_wy = DimensionImpl.swizzle_wy;
        pub const swizzle_wz = DimensionImpl.swizzle_wz;
        pub const swizzle_ww = DimensionImpl.swizzle_ww;
        // 3D swizzles
        pub const swizzle_xyz = DimensionImpl.swizzle_xyz;
        pub const swizzle_xzy = DimensionImpl.swizzle_xzy;
        pub const swizzle_xyw = DimensionImpl.swizzle_xyw;
        pub const swizzle_xwy = DimensionImpl.swizzle_xwy;
        pub const swizzle_xzw = DimensionImpl.swizzle_xzw;
        pub const swizzle_xwz = DimensionImpl.swizzle_xwz;
        pub const swizzle_yxz = DimensionImpl.swizzle_yxz;
        pub const swizzle_yzx = DimensionImpl.swizzle_yzx;
        pub const swizzle_yxw = DimensionImpl.swizzle_yxw;
        pub const swizzle_ywx = DimensionImpl.swizzle_ywx;
        pub const swizzle_yzw = DimensionImpl.swizzle_yzw;
        pub const swizzle_ywz = DimensionImpl.swizzle_ywz;
        pub const swizzle_zxy = DimensionImpl.swizzle_zxy;
        pub const swizzle_zyx = DimensionImpl.swizzle_zyx;
        pub const swizzle_zxw = DimensionImpl.swizzle_zxw;
        pub const swizzle_zwx = DimensionImpl.swizzle_zwx;
        pub const swizzle_zwy = DimensionImpl.swizzle_zwy;
        pub const swizzle_wzy = DimensionImpl.swizzle_wzy;
        pub const swizzle_wxy = DimensionImpl.swizzle_wxy;
        pub const swizzle_wyx = DimensionImpl.swizzle_wyx;
        pub const swizzle_wxz = DimensionImpl.swizzle_wxz;
        pub const swizzle_wzx = DimensionImpl.swizzle_wzx;
        // 4D swizzles
        pub const swizzle_xyzw = DimensionImpl.swizzle_xyzw;
        pub const swizzle_xzyw = DimensionImpl.swizzle_xzyw;
        pub const swizzle_xwyz = DimensionImpl.swizzle_xwyz;
        pub const swizzle_xwzy = DimensionImpl.swizzle_xwzy;
        pub const swizzle_xzwy = DimensionImpl.swizzle_xzwy;
        pub const swizzle_yxzw = DimensionImpl.swizzle_yxzw;
        pub const swizzle_yzxw = DimensionImpl.swizzle_yzxw;
        pub const swizzle_ywzx = DimensionImpl.swizzle_ywzx;
        pub const swizzle_yzwx = DimensionImpl.swizzle_yzwx;
        pub const swizzle_ywxz = DimensionImpl.swizzle_ywxz;
        pub const swizzle_zxyw = DimensionImpl.swizzle_zxyw;
        pub const swizzle_zyxw = DimensionImpl.swizzle_zyxw;
        pub const swizzle_zwxy = DimensionImpl.swizzle_zwxy;
        pub const swizzle_zwyx = DimensionImpl.swizzle_zwyx;
        pub const swizzle_zywx = DimensionImpl.swizzle_zywx;
        pub const swizzle_zxwy = DimensionImpl.swizzle_zxwy;
        pub const swizzle_wxyz = DimensionImpl.swizzle_wxyz;
        pub const swizzle_wzyx = DimensionImpl.swizzle_wzyx;
        pub const swizzle_wyxz = DimensionImpl.swizzle_wyxz;
        pub const swizzle_wxzy = DimensionImpl.swizzle_wxzy;
        pub const swizzle_wyzx = DimensionImpl.swizzle_wyzx;
        pub const swizzle_wzxy = DimensionImpl.swizzle_wzxy;
    };
}

pub const Vec2 = Vec(2, f32);
pub const Vec3 = Vec(3, f32);
pub const Vec4 = Vec(4, f32);

test "vector" {
    _ = std.testing.expect;
    const expect_approx = std.testing.expectApproxEqAbs;
    {
        logger.println("Conversion: v4 -> v3: ");
        const v4 = Vec4.init_dupl(2.0);
        const v3: Vec3 = v4.to_vec3();
        v3.print();
    }
    {
        logger.println("Lerp: ");
        const start: Vec4 = .init(3, 2, 3, 4);
        const end: Vec4 = .init(6, 1, 5, 10);
        const lerp = start.lerp(end, 0.5);
        lerp.print();
    }
    {
        logger.println("Reciprocal: ");
        var v: Vec4 = .init(3, 2, 3, 4);
        v.reciprocal_inplace();
        v.print();
    }
    {
        logger.println("Normalization: ");
        var v: Vec3 = .init(0, 1, 3);
        v.normalize_inplace();
        v.print();
        logger.printfln("magnitude is: {}", .{v.magnitude()});
        try expect_approx(1.0, v.magnitude(), math.floatEps(f32));
    }
    {
        logger.println("Negation: ");
        var v: Vec4 = .init(3, 2, 3, 4);
        v.negate_inplace();
        v.print();
    }
    {
        const v1: Vec3 = .right;
        const v2: Vec3 = .left;
        const dot_v = v1.dot(v2);
        logger.printfln("Dot product = {d}", .{dot_v});
        try expect_approx(-1, dot_v, math.floatEps(f32));
    }
    {
        logger.println("Cross product: ");
        const v1: Vec3 = .right;
        const v2: Vec3 = .bottom;
        const cross_v: Vec3 = v1.cross(v2);
        cross_v.print();
    }
    {
        logger.println("Swizzling");
        const v1: Vec4 = .init(1, 2, 3, 4);
        logger.println("wzxy: ");
        v1.swizzle_wzxy().print();
        logger.println("ywz: ");
        v1.swizzle_ywz().print();
        logger.println("xy: ");
        v1.swizzle_xy().print();
    }
}
