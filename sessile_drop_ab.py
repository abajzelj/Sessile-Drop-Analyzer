import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
import os
from datetime import datetime

# ==========================================
# 1. PHYSICS (LAPLACE)
# ==========================================
def laplace_ode(s, y, beta):
    x, z, phi = y
    if x < 1e-6: term = 1.0 
    else: term = np.sin(phi) / x
    return [np.cos(phi), np.sin(phi), 2 + beta * z - term]

def solve_profile(beta, R0, s_max=5.0):
    ds = 1e-4
    sol = solve_ivp(laplace_ode, [ds, s_max], [ds, 0, ds], args=(beta,), 
                    dense_output=True, rtol=1e-6, atol=1e-8)
    s_eval = np.linspace(ds, s_max, 500)
    res = sol.sol(s_eval)
    return res[0] * R0, res[1] * R0

def get_volume_px(x_arr, z_arr):
    order = np.argsort(z_arr)
    return np.abs(np.pi * np.trapz(x_arr[order]**2, z_arr[order]))

# ==========================================
# 2. POINT SELECTION
# ==========================================
class GeometrySelector:
    def __init__(self, img, contour_points):
        self.img = img
        self.contour_points = contour_points 
        
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.2)
        
        self.ax.imshow(img, cmap='gray')
        self.title_base = "Procedure: 1. Zoom -> 2. Click 'Capture' button -> 3. Click on the image"
        self.ax.set_title(self.title_base)
        self.ax.plot(contour_points[:, 0], contour_points[:, 1], 'r.', markersize=2)
        
        self.clicks = []
        self.selected_segment = None
        self.baseline_angle = 0.0
        self.contact_pts = [] 
        
        self.waiting_for_pick = False
        
        ax_btn = plt.axes([0.4, 0.05, 0.2, 0.075])
        self.btn_pick = Button(ax_btn, 'CAPTURE POINT (Start)', color='lightblue', hovercolor='0.975')
        self.btn_pick.on_clicked(self.enable_picking)
        
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        
        print("\nSELECTION INSTRUCTIONS:")
        print("1. Use the Zoom tool to get closer to the contact point.")
        print("2. Click the 'CAPTURE POINT' button below the image.")
        print("3. Click on the contact point in the image.")
        print("4. Repeat for the right side and the arc.")
        
        plt.show()

    def enable_picking(self, event):
        if len(self.clicks) >= 3:
            return
            
        self.waiting_for_pick = True
        
        step = len(self.clicks)
        msg = ""
        if step == 0: msg = "LEFT CONTACT"
        elif step == 1: msg = "RIGHT CONTACT"
        elif step == 2: msg = "POINT ON ARC"
        
        self.ax.set_title(f"WAITING FOR CLICK ON IMAGE: Mark {msg}", color='red', fontweight='bold')
        self.btn_pick.label.set_text(f"Click on {msg}...")
        self.btn_pick.color = 'orange'
        self.fig.canvas.draw()

    def onclick(self, event):
        if not self.waiting_for_pick: return
        if event.inaxes != self.ax: return
        if self.fig.canvas.toolbar.mode != '': return 
        
        click_xy = np.array([event.xdata, event.ydata])
        dists = np.linalg.norm(self.contour_points - click_xy, axis=1)
        idx = np.argmin(dists)
        pt = self.contour_points[idx]
        
        self.clicks.append((idx, pt))
        
        lbl = str(len(self.clicks))
        info = ""
        if len(self.clicks) == 1: info = " (L)"
        elif len(self.clicks) == 2: info = " (R)"
        elif len(self.clicks) == 3: info = " (Arc)"
        
        self.ax.plot(pt[0], pt[1], 'bo', markersize=8)
        self.ax.text(pt[0], pt[1], lbl + info, color='yellow', fontweight='bold')
        
        self.waiting_for_pick = False
        self.ax.set_title(self.title_base, color='black')
        
        if len(self.clicks) < 3:
            next_step = "RIGHT CONTACT" if len(self.clicks) == 1 else "POINT ON ARC"
            self.btn_pick.label.set_text(f"Capture: {next_step}")
            self.btn_pick.color = 'lightblue'
        else:
            self.btn_pick.label.set_text("Done!")
            self.btn_pick.color = 'lightgreen'
            self.process() 
            
        self.fig.canvas.draw()

    def process(self):
        pts_contact = [self.clicks[0], self.clicks[1]]
        pts_contact.sort(key=lambda x: x[1][0])
        
        idx_L, pt_L = pts_contact[0]
        idx_R, pt_R = pts_contact[1]
        _, pt_Mid = self.clicks[2]
        
        self.contact_pts = [pt_L, pt_R]
        
        dx = pt_R[0] - pt_L[0]
        dy = pt_R[1] - pt_L[1]
        self.baseline_angle = np.arctan2(dy, dx)
        
        pts = self.contour_points
        
        if idx_L < idx_R:
            seg1 = pts[idx_L : idx_R+1]
        else:
            seg1 = np.vstack((pts[idx_L:], pts[:idx_R+1]))
            
        if idx_R < idx_L:
            seg2 = pts[idx_R : idx_L+1]
        else:
            seg2 = np.vstack((pts[idx_R:], pts[:idx_L+1]))
            
        dist1 = np.min(np.linalg.norm(seg1 - pt_Mid, axis=1))
        dist2 = np.min(np.linalg.norm(seg2 - pt_Mid, axis=1))
        
        self.selected_segment = seg1 if dist1 < dist2 else seg2
        
        self.ax.plot(self.selected_segment[:,0], self.selected_segment[:,1], 'g-', lw=2)
        self.ax.set_title("Segment selected! Close the window to calculate...")
        self.fig.canvas.draw()

# ==========================================
# 3. MAIN LOGIC
# ==========================================
class DropAnalyzer:
    def __init__(self, path, vol, rho):
        self.path = path
        self.img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if self.img is None: raise ValueError("No image")
        self.vol_target = vol
        self.rho = rho
        self.g = 9.81
        self.pts_raw = None
        self.pts_final = None
        self.angle = 0
        self.contact_pts = None

    def step1_edge(self):
        print("Select threshold...")
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)
        
        ax.imshow(self.img, cmap='gray')
        ax.set_title("Use Zoom, then move the slider.")
        
        line_contour, = ax.plot([], [], 'r.', ms=1)
        
        ax_sl = plt.axes([0.2, 0.1, 0.6, 0.03])
        sl = Slider(ax_sl, 'Th', 0, 255, valinit=210)
        
        def update(val):
            _, b = cv2.threshold(self.img, int(sl.val), 255, cv2.THRESH_BINARY)
            c, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if c:
                best = max(c, key=cv2.contourArea)
                self.pts_raw = best[::2, 0, :] 
                line_contour.set_data(self.pts_raw[:, 0], self.pts_raw[:, 1])
            else:
                line_contour.set_data([], [])
            
            fig.canvas.draw_idle()
            
        sl.on_changed(update)
        update(0)
        plt.show()

    def step2_select(self):
        sel = GeometrySelector(self.img, self.pts_raw)
        self.pts_final = sel.selected_segment
        self.angle = sel.baseline_angle
        self.contact_pts = sel.contact_pts

    def find_apex_robust(self, pts_rot):
        min_y_idx = np.argmin(pts_rot[:, 1])
        y_min_val = pts_rot[min_y_idx, 1]
        height_threshold = y_min_val + 20
        
        mask_top = pts_rot[:, 1] < height_threshold
        pts_top = pts_rot[mask_top]
        
        if len(pts_top) < 5:
            return pts_rot[min_y_idx, 0], pts_rot[min_y_idx, 1]
            
        try:
            coeffs = np.polyfit(pts_top[:, 0], pts_top[:, 1], 2)
            a, b, c = coeffs
            apex_x = -b / (2*a)
            apex_y = np.polyval(coeffs, apex_x)
            return apex_x, apex_y
        except:
            return pts_rot[min_y_idx, 0], pts_rot[min_y_idx, 1]

    def fit_side(self, X_exp, Z_exp, side_name):
        print(f"  -> Calculating {side_name} side...")
        if len(X_exp) < 10 or len(Z_exp) < 10:
            return (1.0, 100.0), 9999.0
        
        h = np.max(Z_exp)
        w = np.max(X_exp)
        
        def objective(params):
            beta, R0 = params
            if beta < 0.05 or beta > 100: return 1e6
            if R0 < 5 or R0 > w*2.5: return 1e6
            try:
                x_t, z_t = solve_profile(beta, R0, s_max=6.0)
                if z_t.max() < h: return 1e6 + (h - z_t.max())*1000
                x_int = np.interp(Z_exp, z_t, x_t)
                return np.mean((X_exp - x_int)**2)
            except: return 1e6

        bounds = [(0.1, 5.0), (w/4, w*2)] 
        res = differential_evolution(objective, bounds, strategy='best1bin', 
                                     popsize=10, tol=0.01, mutation=(0.5, 1))
        return res.x, res.fun

    def save_results(self, h_mm, d_max, d_contact, gamma_L, gamma_R, gamma_avg):
        try:
            folder_path = os.path.dirname(os.path.abspath(self.path))
            base_name = os.path.splitext(os.path.basename(self.path))[0]
            result_file = os.path.join(folder_path, f"{base_name}.txt")
            
            img_name_full = os.path.basename(self.path)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            lines = [
                f"--------------------------------------------------",
                f"MEASUREMENT DATE: {timestamp}",
                f"FILE:             {img_name_full}",
                f"INPUT DATA:",
                f"  Volume:         {self.vol_target} mm3",
                f"  Delta Rho:      {self.rho} kg/m3",
                f"RESULTS:",
                f"  Height:            {h_mm:.4f} mm",
                f"  Max diameter:      {d_max:.4f} mm",
                f"  Contact diameter:  {d_contact:.4f} mm",
                f"  Gamma (Left):      {'/' if gamma_L is None else f'{gamma_L:.2f}'} mN/m",
                f"  Gamma (Right):     {'/' if gamma_R is None else f'{gamma_R:.2f}'} mN/m",
                f"  Gamma (AVERAGE):   {gamma_avg:.2f} mN/m",
                f"--------------------------------------------------\n"
            ]
            
            with open(result_file, "a", encoding="utf-8") as f:
                f.write("\n".join(lines))
                
            print(f"\n[INFO] Data successfully saved to: {result_file}")
            
        except Exception as e:
            print(f"\n[ERROR] Failed to save data: {e}")

    def step3_calc(self):
        if self.pts_final is None: return
        
        pts = self.pts_final
        p_anchor = self.contact_pts[0]
        a = -self.angle
        R_mat = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        
        pts_centered = pts - p_anchor
        pts_rot = np.dot(pts_centered, R_mat.T)
        
        apex_x, apex_y = self.find_apex_robust(pts_rot)
        
        X_rel = pts_rot[:, 0] - apex_x
        Z_rel = pts_rot[:, 1] - apex_y
        
        mask = Z_rel >= 0
        X_rel = X_rel[mask]
        Z_rel = Z_rel[mask]
        
        mask_L = X_rel < 0
        mask_R = X_rel > 0
        
        XL, ZL = np.abs(X_rel[mask_L]), Z_rel[mask_L]
        idx_L = np.argsort(ZL)
        XL, ZL = XL[idx_L], ZL[idx_L]
        
        XR, ZR = np.abs(X_rel[mask_R]), Z_rel[mask_R]
        idx_R = np.argsort(ZR)
        XR, ZR = XR[idx_R], ZR[idx_R]
        
        print("\nStarting optimization...")
        (beta_L, R0_L), err_L = self.fit_side(XL, ZL, "LEFT")
        (beta_R, R0_R), err_R = self.fit_side(XR, ZR, "RIGHT")
        
        valid_L = (err_L < 1000)
        valid_R = (err_R < 1000)
        
        if not valid_L and not valid_R: return
        
        h_px = np.max(Z_rel)

        vals_b, vals_r = [], []
        
        if valid_L: 
            vals_b.append(beta_L); vals_r.append(R0_L)
            
        if valid_R: 
            vals_b.append(beta_R); vals_r.append(R0_R)
            
        beta_avg = np.mean(vals_b)
        R0_avg = np.mean(vals_r)
        
        xt, zt = solve_profile(beta_avg, R0_avg, s_max=10.0)
        mask_v = zt <= h_px
        vol_px = get_volume_px(xt[mask_v], zt[mask_v])
        
        scale = (self.vol_target / vol_px)**(1/3) 
        
        h_mm = h_px * scale
        x_prof = xt[mask_v]
        
        if len(x_prof) > 0:
            d_contact_mm = 2 * x_prof[-1] * scale
            d_max_mm = 2 * np.max(x_prof) * scale
        else:
            d_contact_mm = 0
            d_max_mm = 0

        def calc_gamma(b, r_px):
            r_m = r_px * scale / 1000.0 
            return (self.rho * self.g * r_m**2) / b * 1000 
            
        print("\n" + "="*40)
        print(f"RESULTS")
        print(f"Drop dimensions:")
        print(f"scale: {1/scale:.4f}")
        print(f"  Height:            {h_mm:.3f} mm")
        print(f"  Max diameter:      {d_max_mm:.3f} mm")
        print(f"  Contact diameter:  {d_contact_mm:.3f} mm")
        print("-" * 40)
        print(f"Surface tension (Gamma):")
        
        gamma_final = 0
        cnt = 0
        gL_val = None
        gR_val = None
        
        if valid_L:
            gL_val = calc_gamma(beta_L, R0_L)
            print(f"  Left side:   {gL_val:.2f} mN/m")
            gamma_final += gL_val; cnt += 1
        if valid_R:
            gR_val = calc_gamma(beta_R, R0_R)
            print(f"  Right side:  {gR_val:.2f} mN/m")
            gamma_final += gR_val; cnt += 1
            
        if cnt > 0:
            gamma_final /= cnt
            print(f"  AVERAGE:     {gamma_final:.2f} mN/m")
        print("="*40)
        
        self.save_results(h_mm, d_max_mm, d_contact_mm, gL_val, gR_val, gamma_final)
        
        # Prepare data for interactive plot
        plot_data = {
            'apex': (apex_x, apex_y),
            'anchor': p_anchor,
            'angle': self.angle,
            'scale': scale,
            'h_limit': h_px,
            'L': (beta_L, R0_L) if valid_L else None,
            'R': (beta_R, R0_R) if valid_R else None,
            'gamma': gamma_final
        }
        
        self.plot_final_interactive(plot_data)

    def plot_final_interactive(self, data):
        fig, ax = plt.subplots(figsize=(14, 10))
        plt.subplots_adjust(right=0.8) # Space for buttons on the right
        
        ax.imshow(self.img, cmap='gray')
        
        ax_rot, ay_rot = data['apex']
        p_anchor = data['anchor']
        ang = data['angle']
        h_limit = data['h_limit']
        scale = data['scale']
        
        # Dictionary for objects we will toggle
        plot_objects = {
            'fit': [],
            'raw': [],
            'text': []
        }
        
        # Rotation function
        def rotate_point(px, py):
            pts_c = np.column_stack((px, py))
            Rb = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
            return np.dot(pts_c, Rb) + p_anchor

        # 1. RAW MEASUREMENT
        l_raw, = ax.plot(self.pts_final[:,0], self.pts_final[:,1], 'k.', ms=1, alpha=0.3, label='Measurement')
        plot_objects['raw'].append(l_raw)

        # 2. FITTED CURVES
        def draw_side(params, side_sign, color):
            beta, R0 = params
            
            # A) Fitted curve
            xt, zt = solve_profile(beta, R0, s_max=12.0)
            idx_over = np.where(zt > h_limit)[0]
            if len(idx_over) > 0:
                cut_idx = idx_over[0]
                xt = xt[:cut_idx+1]
                zt = zt[:cut_idx+1]
            
            X_mod = side_sign * xt
            Z_mod = zt
            X_rot = X_mod + ax_rot
            Y_rot = Z_mod + ay_rot
            pts_final = rotate_point(X_rot, Y_rot)
            
            l_fit, = ax.plot(pts_final[:,0], pts_final[:,1], color=color, lw=2)
            plot_objects['fit'].append(l_fit)

        if data['L']: draw_side(data['L'], -1, 'blue')
        if data['R']: draw_side(data['R'], 1, 'red')

        # 3. SCALE AND TITLE
        w_px = 1.0 / scale
        rect = plt.Rectangle((50, 50), w_px, 15, color='yellow')
        ax.add_patch(rect)
        t_scale = ax.text(50, 40, "1 mm", color='yellow', fontsize=12, fontweight='bold')
        plot_objects['text'].append(rect)
        plot_objects['text'].append(t_scale)
        
        file_label = os.path.basename(self.path)
        ax.set_title(f"{file_label} | Gamma: {data['gamma']:.1f} mN/m")

        # --- INTERACTIVE CHECKBOXES ---
        ax_check = plt.axes([0.82, 0.4, 0.15, 0.2]) # [left, bottom, width, height]
        labels = ["Fitted model", "Measurement", "Results"]
        visibility = [True, True, True]
        check = CheckButtons(ax_check, labels, visibility)

        def func(label):
            key = ""
            if label == "Fitted model": key = 'fit'
            elif label == "Measurement": key = 'raw'
            elif label == "Results": key = 'text'
            
            # Toggle visibility
            for artist in plot_objects[key]:
                artist.set_visible(not artist.get_visible())
            plt.draw()

        check.on_clicked(func)
        plt.show()

if __name__ == "__main__":
    try:
        path = input("Image: ").strip().replace('"','')
        v_in = input("Volume [mm3] (50): ")
        vol = float(v_in) if v_in else 50.0
        r_in = input("Delta Rho [kg/m3] (7000): ")
        rho = float(r_in) if r_in else 7000.0
        
        app = DropAnalyzer(path, vol, rho)
        app.step1_edge()
        app.step2_select()
        app.step3_calc()
        
    except Exception as e:
        import traceback
        traceback.print_exc()
    input("Press Enter to exit...")