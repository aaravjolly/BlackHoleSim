import moderngl
import moderngl_window as mglw
import numpy as np
import math

"""
Black Hole Simulator — Schwarzschild Metric

Uses the proper geodesic equation for null rays in Schwarzschild spacetime.
Working in geometrized units where G = c = 1, with mass parameter M.
Schwarzschild radius rs = 2M.

The geodesic equation in Schwarzschild coordinates (converted to Cartesian
for 3D ray marching) gives the acceleration:

    d²xⁱ/dλ² = -Γⁱ_μν (dxᵘ/dλ)(dxᵛ/dλ)

For a photon in Cartesian-like Schwarzschild coordinates, this reduces to:

    a = -(rs/(2r³)) * [ x(c² - 2v·v) + 2v(x·v)(1 + rs/(2(r-rs))) ]

    where the key GR term (rs/(r-rs)) diverges at the horizon and creates
    the photon sphere at r = 1.5*rs.

Simplified for numerical integration (Verlet), the effective acceleration is:

    a = -(rs/2) * [ x/r³ - 3(x·v)²·x/r⁵ + ... ]

We use the standard form from Müller & Grave (2009) for null geodesics:

    d²x/dλ² = -rs/(2r³) * x + (3*rs)/(2r⁵) * (x·x - (x·v)²/(v·v)) * ... 




"""



class Camera:
    def __init__(self):
        self.radius = 55.0
        self.azimuth = math.pi / 2.0
        self.elevation = math.pi / 2.8

    def position(self):
        return np.array([
            self.radius * math.sin(self.elevation) * math.cos(self.azimuth),
            self.radius * math.cos(self.elevation),
            self.radius * math.sin(self.elevation) * math.sin(self.azimuth),
        ], dtype="f4")

    def forward(self):
        pos = self.position()
        f = -pos / np.linalg.norm(pos)
        return f.astype("f4")

    def right(self):
        f = self.forward()
        up = np.array([0, 1, 0], dtype="f4")
        r = np.cross(f, up)
        norm = np.linalg.norm(r)
        if norm < 1e-6:
            up = np.array([0, 0, 1], dtype="f4")
            r = np.cross(f, up)
        r /= np.linalg.norm(r)
        return r.astype("f4")

    def up(self):
        f = self.forward()
        r = self.right()
        u = np.cross(r, f)
        u /= np.linalg.norm(u)
        return u.astype("f4")



def schwarzschild_accel(pos, vel, rs):
    """
    Geodesic acceleration for a null ray in Schwarzschild spacetime.

    a = -(rs/2) * x/r³ * (1 + 3·|x × v|²/r²)

    This is derived from the Schwarzschild Christoffel symbols in
    quasi-Cartesian coordinates. The factor (1 + 3h²/r²) is the
    GR correction that creates the photon sphere at r = 1.5·rs.
    """
    r = np.linalg.norm(pos)
    if r < rs * 0.5:
        return np.zeros(3, dtype='f4')

    r2 = r * r
    r3 = r2 * r

    # Angular momentum squared: |x × v|²
    cross = np.cross(pos, vel)
    h2 = np.dot(cross, cross)

    # GR correction factor
    gr_factor = 1.0 + 3.0 * h2 / r2

    # Full Schwarzschild geodesic acceleration
    acc = -(rs / 2.0) * pos / r3 * gr_factor

    return acc.astype('f4')



def compute_light_rays(rs, num_rays=50, y_offset=0.0):
    rays = []
    b_values = np.linspace(-rs * 14.0, rs * 14.0, num_rays)
    start_x = 80.0

    for b in b_values:
        pos = np.array([start_x, y_offset, b], dtype='f4')
        vel = np.array([-1.0, 0.0, 0.0], dtype='f4')

        path = [pos.copy()]
        absorbed = False

        for step in range(10000):
            r = np.linalg.norm(pos)

            # Event horizon check
            if r < rs * 1.01:
                absorbed = True
                for k in range(10):
                    frac = (k + 1) / 10.0
                    path.append(pos * (1.0 - frac))
                break

            if r > 120.0 and step > 50:
                break

            # Adaptive step size: very fine near photon sphere
            if r < rs * 2.0:
                dt = 0.01
            elif r < rs * 4.0:
                dt = max(0.02, 0.005 * r)
            else:
                dt = max(0.03, min(0.2, 0.015 * r))

            # Velocity Verlet integration with Schwarzschild acceleration
            acc = schwarzschild_accel(pos, vel, rs)

            # Half-step velocity
            vel_half = vel + acc * dt * 0.5
            # Full-step position
            pos_new = pos + vel_half * dt
            # Recalculate acceleration at new position
            acc_new = schwarzschild_accel(pos_new, vel_half, rs)
            # Full-step velocity
            vel = vel_half + acc_new * dt * 0.5

            # Renormalize velocity (photons travel at c=1)
            speed = np.linalg.norm(vel)
            if speed > 0:
                vel = vel / speed

            pos = pos_new
            path.append(pos.copy())

        rays.append((np.array(path, dtype='f4'), absorbed))

    return rays


def build_ribbon(path, width=0.3):
    """
    Turn a path (Nx3) into a triangle-strip ribbon.
    Perpendicular direction computed from path tangent × Y-up.
    """
    n = len(path)
    if n < 2:
        return None

    verts = []
    for i in range(n):
        if i == 0:
            tang = path[1] - path[0]
        elif i == n - 1:
            tang = path[-1] - path[-2]
        else:
            tang = path[i + 1] - path[i - 1]

        tang_len = np.linalg.norm(tang)
        if tang_len < 1e-8:
            tang = np.array([1, 0, 0], dtype='f4')
        else:
            tang = tang / tang_len

        up = np.array([0, 1, 0], dtype='f4')
        perp = np.cross(tang, up)
        perp_len = np.linalg.norm(perp)
        if perp_len < 1e-8:
            perp = np.array([1, 0, 0], dtype='f4')
        else:
            perp = perp / perp_len

        p = path[i]
        idx = float(i)

        v1 = p + perp * width * 0.5
        v2 = p - perp * width * 0.5

        verts.append([v1[0], v1[1], v1[2], idx, 1.0, 0.0])
        verts.append([v2[0], v2[1], v2[2], idx, -1.0, 0.0])

    return np.array(verts, dtype='f4')



class BlackHoleEngine(mglw.WindowConfig):

    gl_version = (3, 3)
    window_size = (1000, 800)
    aspect_ratio = None
    title = "Black Hole — Schwarzschild Geodesics"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ctx.enable(moderngl.DEPTH_TEST)

        self.camera = Camera()

        self.M = 2.0
        self.rs = 2 * self.M  # Schwarzschild radius

        self.build_ray_program()
        self.build_quad()
        self.build_star()
        self.build_light_rays()

    
    def build_ray_program(self):
        self.ray_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                out vec2 uv;
                void main() {
                    uv = in_pos * 0.5 + 0.5;
                    gl_Position = vec4(in_pos, 0, 1);
                }
            """,
            fragment_shader="""
                #version 330
                in vec2 uv;
                out vec4 fragColor;

                uniform vec3 camPos;
                uniform vec3 camForward;
                uniform vec3 camRight;
                uniform vec3 camUp;
                uniform float rs;
                uniform float aspect;
                uniform float time;

                // Schwarzschild geodesic acceleration for null rays
                // a = -(rs/2) * x/r^3 * (1 + 3*h^2/r^2)
                // where h^2 = |x cross v|^2
                vec3 geodesic_accel(vec3 x, vec3 v, float rs_val) {
                    float r = length(x);
                    if (r < rs_val * 0.5) return vec3(0.0);

                    float r2 = r * r;
                    float r3 = r2 * r;

                    // h^2 = |x × v|^2 = r^2*|v|^2 - (x·v)^2
                    vec3 h = cross(x, v);
                    float h2 = dot(h, h);

                    // GR correction: this creates the photon sphere at r = 1.5*rs
                    float gr_factor = 1.0 + 3.0 * h2 / r2;

                    return -(rs_val / 2.0) * x / r3 * gr_factor;
                }

                vec3 disk_color(float r, float angle) {
                    // ISCO (innermost stable circular orbit) at 3*rs for Schwarzschild
                    float inner = rs * 3.0;
                    float outer = rs * 8.0;
                    float t = clamp((r - inner) / (outer - inner), 0.0, 1.0);

                    vec3 hot = vec3(1.0, 0.85, 0.5);
                    vec3 warm = vec3(1.0, 0.5, 0.1);
                    vec3 cool = vec3(0.6, 0.15, 0.02);

                    vec3 col;
                    if (t < 0.3) col = mix(hot, warm, t / 0.3);
                    else col = mix(warm, cool, (t - 0.3) / 0.7);

                    float brightness = pow(1.0 - t, 0.6) * 1.5;

                    // Doppler-like asymmetry (approaching side brighter)
                    float doppler = 1.0 + 0.3 * sin(angle + time * 0.2);

                    float turb = 0.85 + 0.15 * sin(angle * 6.0 + time * 0.5);
                    turb *= 0.9 + 0.1 * sin(r * 2.0 - time * 0.3);

                    return col * brightness * turb * doppler;
                }

                void main() {
                    float fov = 1.2;
                    vec2 screen = uv * 2.0 - 1.0;
                    screen.x *= aspect;

                    vec3 rayDir = normalize(
                        camForward + screen.x * fov * camRight + screen.y * fov * camUp
                    );
                    vec3 pos = camPos;

                    vec3 color = vec3(0.0);
                    bool hit = false;
                    float prevY = pos.y;

                    // Velocity Verlet integration of Schwarzschild geodesics
                    for (int i = 0; i < 1200; i++) {
                        float r = length(pos);

                        // Event horizon
                        if (r < rs) {
                            color = vec3(0.0);
                            hit = true;
                            break;
                        }

                        // Escape
                        if (r > 120.0) break;

                        // Adaptive step: fine near photon sphere (1.5*rs)
                        float stepSize;
                        if (r < rs * 2.0)
                            stepSize = 0.02;
                        else if (r < rs * 4.0)
                            stepSize = clamp(0.005 * r, 0.03, 0.1);
                        else
                            stepSize = clamp(0.02 * r, 0.05, 0.3);

                        // Schwarzschild geodesic acceleration
                        vec3 acc = geodesic_accel(pos, rayDir, rs);

                        // Velocity Verlet
                        vec3 velHalf = rayDir + acc * stepSize * 0.5;
                        vec3 posNew = pos + velHalf * stepSize;
                        vec3 accNew = geodesic_accel(posNew, velHalf, rs);
                        rayDir = velHalf + accNew * stepSize * 0.5;
                        rayDir = normalize(rayDir);
                        pos = posNew;

                        // Disk crossing check
                        float curY = pos.y;
                        if (prevY * curY < 0.0 || abs(curY) < 0.06) {
                            float diskR = length(vec2(pos.x, pos.z));
                            // Disk from ISCO (3*rs) to 8*rs
                            if (diskR > rs * 3.0 && diskR < rs * 8.0) {
                                float angle = atan(pos.z, pos.x);
                                color = disk_color(diskR, angle);
                                hit = true;
                                break;
                            }
                        }
                        prevY = curY;
                    }

                    // Starfield background
                    if (!hit) {
                        vec3 rd = normalize(pos - camPos);
                        float n = fract(sin(dot(floor(rd * 300.0),
                            vec3(12.9898, 78.233, 45.164))) * 43758.5453);
                        if (n > 0.997) {
                            float stars = (n - 0.997) / 0.003;
                            stars *= stars;
                            color = vec3(stars * 0.8, stars * 0.85, stars * 1.0);
                        }
                    }

                    // Tone mapping
                    color = color / (color + vec3(1.0));
                    fragColor = vec4(color, 1.0);
                }
            """
        )

    def build_quad(self):
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype='f4')
        self.quad_vbo = self.ctx.buffer(quad.tobytes())
        self.quad_vao = self.ctx.simple_vertex_array(self.ray_prog, self.quad_vbo, "in_pos")

   
    def build_star(self):
        self.star_pos = np.array([80.0, 0.0, 0.0], dtype='f4')

        self.star_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                uniform mat4 mvp;
                uniform vec3 starPos;
                uniform vec3 camRight;
                uniform vec3 camUp;
                in vec2 in_pos;
                out vec2 vUV;
                void main() {
                    float size = 4.0;
                    vec3 worldPos = starPos
                        + camRight * in_pos.x * size
                        + camUp * in_pos.y * size;
                    vUV = in_pos;
                    gl_Position = mvp * vec4(worldPos, 1);
                }
            """,
            fragment_shader="""
                #version 330
                in vec2 vUV;
                out vec4 fragColor;
                void main() {
                    float d = length(vUV);
                    float core = exp(-d * d * 6.0);
                    float glow = exp(-d * d * 1.0) * 0.6;
                    float intensity = core + glow;
                    vec3 col = mix(vec3(1.0, 0.8, 0.4), vec3(1.0, 1.0, 0.9), core);
                    fragColor = vec4(col * intensity, intensity);
                }
            """
        )

        star_quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype='f4')
        self.star_vbo = self.ctx.buffer(star_quad.tobytes())
        self.star_vao = self.ctx.simple_vertex_array(self.star_prog, self.star_vbo, "in_pos")

    
    def build_light_rays(self):
        all_rays = compute_light_rays(self.rs, num_rays=50, y_offset=0.0)
        all_rays += compute_light_rays(self.rs, num_rays=16, y_offset=1.2)
        all_rays += compute_light_rays(self.rs, num_rays=16, y_offset=-1.2)

        self.photon_rays = []

        self.light_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                uniform mat4 mvp;
                in vec3 in_pos;
                in float in_idx;
                in float in_side;
                out float vIdx;
                out float vSide;
                void main() {
                    vIdx = in_idx;
                    vSide = in_side;
                    gl_Position = mvp * vec4(in_pos, 1);
                }
            """,
            fragment_shader="""
                #version 330
                in float vIdx;
                in float vSide;
                out vec4 fragColor;

                uniform float headIdx;
                uniform float tailLen;

                void main() {
                    float behind = headIdx - vIdx;

                    if (behind < 0.0 || behind > tailLen) {
                        discard;
                    }

                    float t = 1.0 - behind / tailLen;

                    float edgeFade = 1.0 - abs(vSide) * 0.3;
                    float brightness = (0.4 + 0.6 * t) * edgeFade;

                    vec3 col = vec3(1.0, 0.95, 0.85) * brightness * 1.5;
                    float alpha = (0.3 + 0.7 * t * t) * edgeFade;

                    fragColor = vec4(col, alpha);
                }
            """
        )

        total_rays = len(all_rays)
        ribbon_width = 0.35

        for i, (path, absorbed) in enumerate(all_rays):
            if len(path) < 3:
                continue

            ribbon = build_ribbon(path, width=ribbon_width)
            if ribbon is None:
                continue

            n_path = len(path)
            n_verts = len(ribbon)

            vbo = self.ctx.buffer(ribbon.tobytes())
            vao = self.ctx.vertex_array(
                self.light_prog,
                [(vbo, '3f 1f 1f 4x', 'in_pos', 'in_idx', 'in_side')]
            )

            time_offset = (i / total_rays) * 5.0
            self.photon_rays.append((vao, n_path, n_verts, absorbed, time_offset))

    
    def on_render(self, time, frame_time):
        self.ctx.clear(0, 0, 0)

        self.camera.azimuth = math.pi / 2.0 + time * 0.03

        cam_pos = self.camera.position()
        cam_fwd = self.camera.forward()
        cam_right = self.camera.right()
        cam_up = self.camera.up()

        
        self.ray_prog["camPos"].value = tuple(cam_pos)
        self.ray_prog["camForward"].value = tuple(cam_fwd)
        self.ray_prog["camRight"].value = tuple(cam_right)
        self.ray_prog["camUp"].value = tuple(cam_up)
        self.ray_prog["rs"].value = self.rs
        self.ray_prog["aspect"].value = self.wnd.aspect_ratio
        self.ray_prog["time"].value = time

        self.ctx.disable(moderngl.DEPTH_TEST)
        self.quad_vao.render(moderngl.TRIANGLE_STRIP)

       
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)

        proj = self.perspective(45, self.wnd.aspect_ratio, 0.1, 200)
        view = self.look_at(cam_pos, (0, 0, 0), (0, 1, 0))
        mvp = proj @ view

        
        self.star_prog["mvp"].write(mvp.astype("f4").tobytes())
        self.star_prog["starPos"].value = tuple(self.star_pos)
        self.star_prog["camRight"].value = tuple(cam_right)
        self.star_prog["camUp"].value = tuple(cam_up)
        self.star_vao.render(moderngl.TRIANGLE_STRIP)

        
        self.light_prog["mvp"].write(mvp.astype("f4").tobytes())

        photon_speed = 100.0
        tail_length = 120.0

        for vao, n_path, n_verts, absorbed, t_offset in self.photon_rays:
            cycle_time = (n_path + tail_length) / photon_speed
            local_time = (time - t_offset) % cycle_time
            head = local_time * photon_speed

            self.light_prog["headIdx"].value = head
            self.light_prog["tailLen"].value = tail_length

            vao.render(moderngl.TRIANGLE_STRIP)

        self.ctx.disable(moderngl.BLEND)


    def perspective(self, fov, aspect, near, far):
        f = 1 / math.tan(math.radians(fov) / 2)
        return np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ], dtype='f4')

    def look_at(self, eye, target, up):
        eye = np.array(eye, dtype='f4')
        target = np.array(target, dtype='f4')
        up = np.array(up, dtype='f4')
        f = (target - eye); f /= np.linalg.norm(f)
        s = np.cross(f, up); s /= np.linalg.norm(s)
        u = np.cross(s, f)
        m = np.identity(4, dtype='f4')
        m[0, :3] = s; m[1, :3] = u; m[2, :3] = -f
        trans = np.identity(4, dtype='f4')
        trans[:3, 3] = -eye
        return m @ trans


if __name__ == "__main__":
    mglw.run_window_config(BlackHoleEngine)
