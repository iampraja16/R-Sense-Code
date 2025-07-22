from flask import Flask, Response, jsonify, render_template, request
import threading, queue, json, time
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import requests
import joblib
from scipy.signal import welch
from scipy.stats import kurtosis, entropy
import paho.mqtt.client as mqtt
import urllib3
from scipy.signal import butter, filtfilt
import random
import math
import pywt

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# endpoint alpi
WEBHOOK_URL = "https://entrance-pig-batteries-dns.trycloudflare.com/api/devices/data/2e4693ec-5440-4c96-8c9d-602a6c468ddd"

app = Flask(__name__)

class RoadMonitor:
    FEATURE_NAMES = [
        f"{axis}_{feat}" for axis in ['vib','x','y','z'] for feat in [
            "rms", "peak_to_peak", "max_val", "min_val", "mean_val", "ten_point_avg", "kurtosis",
            "band_0.0_0.5_power", "band_0.0_0.5_rms", "band_0.0_0.5_max",
            "band_0.5_1.0_power", "band_0.5_1.0_rms", "band_0.5_1.0_max",
            "band_1.0_1.5_power", "band_1.0_1.5_rms", "band_1.0_1.5_max",
            "band_1.5_2.0_power", "band_1.5_2.0_rms", "band_1.5_2.0_max",
            "band_2.0_2.5_power", "band_2.0_2.5_rms", "band_2.0_2.5_max",
            "wavelet_morl_s4_rms", "wavelet_morl_s4_ten_pt_avg",
            "wavelet_morl_s5_rms", "wavelet_morl_s5_ten_pt_avg",
            "wavelet_db6_s4_rms", "wavelet_db6_s4_ten_pt_avg",
            "wavelet_db6_s5_rms", "wavelet_db6_s5_ten_pt_avg",
            "wavelet_db10_s4_rms", "wavelet_db10_s4_ten_pt_avg",
            "wavelet_db10_s5_rms", "wavelet_db10_s5_ten_pt_avg"
        ]
    ]

    def __init__(self, chunk_size=50):
        self.chunk_size   = chunk_size
        self.raw_queue    = queue.Queue()
        self.lock         = threading.Lock()
        self.gps_buffer   = []
        self.gps_lock     = threading.Lock()
        self.clients      = []
        self.last_result  = None
        self.prev_gps = None

        # Load ML model & scaler
        self.model        = joblib.load('./model/rf_model_updated_waveletss.pkl')
        self.scaler       = joblib.load('./model/scaler_updated_waveletss.pkl')
        self.pca          = joblib.load('./model/pca_model_updated_waveletss.pkl')

        # MQTT untuk GPS
        self.mqtt_client  = mqtt.Client()
        self.mqtt_client.on_connect  = self.on_mqtt_connect
        self.mqtt_client.on_message  = self.on_mqtt_message
        self.mqtt_client.connect("mqtt.eclipseprojects.io", 1883, 60)

        # Mulai thread
        threading.Thread(target=self.process_loop, daemon=True).start()
        threading.Thread(target=self.mqtt_client.loop_forever, daemon=True).start()

    def on_mqtt_connect(self, client, userdata, flags, rc):
        client.subscribe("sensor/data/gps")

    def on_mqtt_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            if not payload.get('position.valid', False):
                return
            gps_entry = {
                'timestamp': payload['timestamp'],
                'lat':       payload['position.latitude'],
                'lon':       payload['position.longitude'],
                'alt':       payload['position.altitude'],
            }
            with self.gps_lock:
                self.gps_buffer.append(gps_entry)
                if len(self.gps_buffer) > 100:
                    self.gps_buffer.pop(0)
        except Exception as e:
            print("GPS Error:", e)

    def push_update(self, data):
        for client in list(self.clients):
            try:
                client['queue'].put(data)
            except:
                self.clients.remove(client)

    def find_closest_gps(self, chunk_time):
        with self.gps_lock:
            closest, min_diff = None, float('inf')
            for gps in reversed(self.gps_buffer):
                diff = abs(gps['timestamp'] - chunk_time)
                if diff <= 10 and diff < min_diff:
                    closest, min_diff = gps, diff
            return closest

    # def get_valid_gps(self, gps):
    #     """
    #     Jika GPS kosong (0.0), return lokasi dummy random.
    #     """
    #     if gps['lat'] in [None, 0.0] or gps['lon'] in [None, 0.0]:
    #         # Lokasi random sekitar Jakarta
    #         lat = -6.2 + random.uniform(-0.050, 0.050)
    #         lon = 106.8 + random.uniform(-0.050, 0.050)
    #         return {
    #             'lat': lat,
    #             'lon': lon,
    #             'timestamp': gps['timestamp']
    #         }
    #     return gps

    # List nama fitur urut
    @staticmethod
    def bandpass_filter(data, lowcut=0.1, highcut=0.5, fs=5.0, order=2):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)
    
    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        # Menghitung jarak horizontal (meter) antara dua titik GPS
        R = 6371000  # radius bumi dalam meter
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # def generate_dummy_vibration(self):
    #     # Simulasi data getaran dan sumbu x, y, z
    #     vib = [random.uniform(0.1, 2.0) for _ in range(self.chunk_size)]
    #     x = [random.uniform(-1.0, 1.0) for _ in range(self.chunk_size)]
    #     y = [random.uniform(-1.0, 1.0) for _ in range(self.chunk_size)]
    #     z = [random.uniform(-1.0, 1.0) for _ in range(self.chunk_size)]
    #     total_momentum = [random.uniform(-1.0, 1.0) for _ in range(self.chunk_size)]
    #     total_velocity = [random.uniform(-1.0, 1.0) for _ in range(self.chunk_size)]
    #     full_data = [
    #         {'vibration': vib[i], 'x': x[i], 'y': y[i], 'z': z[i], 'total_momentum': total_momentum[i], 'total_velocity': total_velocity[i]}
    #         for i in range(self.chunk_size)
    #     ]
    #     return {
    #         'vibration': vib,
    #         'x': x,
    #         'y': y,
    #         'z': z,
    #         'total_momentum': total_momentum,
    #         'total_velocity': total_velocity,
    #         'full_data': full_data,
    #         'timestamp': time.time()
    #     }

    # def generate_dummy_gps(self, base_lat=-6.200, base_lon=106.950, base_alt=10.0):
    #     # Random walk di sekitar Cakung, Jakarta Timur
    #     lat = base_lat + random.uniform(-0.005, 0.005)
    #     lon = base_lon + random.uniform(-0.005, 0.005)
    #     alt = base_alt + random.uniform(-2, 2)
    #     return {
    #         'timestamp': time.time(),
    #         'lat': lat,
    #         'lon': lon,
    #         'alt': alt
    #     }

    def calculate_slope_deg(self, prev_gps, curr_gps):
        # prev_gps & curr_gps: dict dengan lat, lon, alt
        if not prev_gps or not curr_gps or 'alt' not in prev_gps or 'alt' not in curr_gps:
            return None
        d_alt = curr_gps.get('alt', 0.0) - prev_gps.get('alt', 0.0)
        d_horiz = self.haversine(prev_gps['lat'], prev_gps['lon'], curr_gps['lat'], curr_gps['lon'])
        if d_horiz == 0:
            return 0.0
        slope_rad = math.atan(d_alt / d_horiz)
        return math.degrees(slope_rad)

    @staticmethod
    def ten_point_average(window):
        if len(window) < 10:
            return 0.0
        sorted_window = np.sort(window)
        avg_peaks = np.mean(sorted_window[-5:])
        avg_valleys = np.mean(sorted_window[:5])
        return np.abs(avg_peaks - avg_valleys)

    def extract_features(self, vib, x, y, z, fs=5):
        try:
            features = {}
            axis_data = {'vib': np.array(vib, dtype=float), 
                         'x': np.array(x, dtype=float), 
                         'y': np.array(y, dtype=float), 
                         'z': np.array(z, dtype=float)
                        #  'total_momentum': np.array(total_momentum, dtype=float), 
                        #  'total_velocity': np.array(total_velocity, dtype=float)
                         }
            
            for axis, data in axis_data.items():
                window = np.array(data)
                if len(window) < 50:
                    window = np.pad(window, (0, 50-len(window)), 'constant')
                # Time domain
                features[f"{axis}_rms"] = float(np.sqrt(np.mean(window**2)))
                features[f"{axis}_peak_to_peak"] = float(np.max(window) - np.min(window))
                features[f"{axis}_max_val"] = float(np.max(window))
                features[f"{axis}_min_val"] = float(np.min(window))
                features[f"{axis}_mean_val"] = float(np.mean(window))
                features[f"{axis}_ten_point_avg"] = float(self.ten_point_average(window))
                features[f"{axis}_kurtosis"] = float(kurtosis(window, nan_policy='omit'))
                # Frequency domain
                freqs, psd = welch(window, fs, nperseg=len(window))
                psd_sum = psd.sum() if psd.sum() else 1.0
                band_edges = np.arange(0, fs/2 + 0.5, 0.5)
                for k in range(len(band_edges) - 1):
                    low_f, high_f = band_edges[k], band_edges[k+1]
                    band_mask = (freqs >= low_f) & (freqs < high_f)
                    current_band_psd = psd[band_mask]
                    band_power = np.sum(current_band_psd) if current_band_psd.size else 0.0
                    band_rms = np.sqrt(np.mean(current_band_psd**2)) if current_band_psd.size else 0.0
                    band_max = np.max(current_band_psd) if current_band_psd.size else 0.0
                    features[f"{axis}_band_{low_f}_{high_f}_power"] = float(band_power)
                    features[f"{axis}_band_{low_f}_{high_f}_rms"] = float(band_rms)
                    features[f"{axis}_band_{low_f}_{high_f}_max"] = float(band_max)
                # Wavelet domain
                wavelets = ['morl', 'db6', 'db10']
                scales = [4, 5]
                for wavelet_name in wavelets:
                    try:
                        if wavelet_name == 'morl':
                            coef, _ = pywt.cwt(window, scales, wavelet_name)
                            for i, scale_val in enumerate(scales):
                                c = coef[i]
                                features[f"{axis}_wavelet_{wavelet_name}_s{scale_val}_rms"] = float(np.sqrt(np.mean(c**2)))
                                features[f"{axis}_wavelet_{wavelet_name}_s{scale_val}_ten_pt_avg"] = float(self.ten_point_average(c))
                        else:
                            max_level = pywt.dwt_max_level(len(window), pywt.Wavelet(wavelet_name).dec_len)
                            for scale_val in scales:
                                if scale_val <= max_level:
                                    coeffs = pywt.wavedec(window, wavelet_name, level=scale_val)
                                    detail_coefs = coeffs[scale_val] if len(coeffs) > scale_val else np.zeros(1)
                                    features[f"{axis}_wavelet_{wavelet_name}_s{scale_val}_rms"] = float(np.sqrt(np.mean(detail_coefs**2)))
                                    features[f"{axis}_wavelet_{wavelet_name}_s{scale_val}_ten_pt_avg"] = float(self.ten_point_average(detail_coefs))
                                else:
                                    features[f"{axis}_wavelet_{wavelet_name}_s{scale_val}_rms"] = 0.0
                                    features[f"{axis}_wavelet_{wavelet_name}_s{scale_val}_ten_pt_avg"] = 0.0
                    except Exception:
                        for scale_val in scales:
                            features[f"{axis}_wavelet_{wavelet_name}_s{scale_val}_rms"] = 0.0
                            features[f"{axis}_wavelet_{wavelet_name}_s{scale_val}_ten_pt_avg"] = 0.0
            return features
        except Exception as e:
            print(f"Error extract_features: {str(e)}")
            return {k: 0.0 for k in RoadMonitor.FEATURE_NAMES}  # Return default
        
    def process_loop(self):
        chunk_count = 0
        while True:
            try:
                raw = self.raw_queue.get(timeout=1)
                # raw = self.generate_dummy_vibration()
                vib_data                 = raw['vibration']
                x_data                   = raw['x']
                z_data                   = raw['z']
                total_momentum_data      = raw['total_momentum']
                total_velocity_data      = raw['total_velocity']
                y_data                   = raw['y']
                full_data                = raw['full_data']
                ntp_time                 = raw['timestamp']
                gps         = self.find_closest_gps(ntp_time) or {
                    'lat': 0.0, 'lon': 0.0, 'alt': 0.0, 'timestamp': ntp_time
                }
                # gps = self.generate_dummy_gps()

                slope_deg = None
                if self.prev_gps:
                    slope_deg = self.calculate_slope_deg(self.prev_gps, gps)
                self.prev_gps = gps

                # Default
                ml_status, stats = 'ML_ERROR', {'max':None,'avg':None}

                # ML extraction & predict
                try:
                    # vib_list = np.array([float(d['vibration']) for d in full_data])  # Convert explicit ke float
                    # x_list = np.array([float(d['x']) for d in full_data])
                    # y_list = np.array([float(d['y']) for d in full_data])
                    # z_list = np.array([float(d['z']) for d in full_data])
                    # total_momentum_list = np.array([float(d['total_momentum']) for d in full_data])
                    # total_velocity_list = np.array([float(d['total_velocity']) for d in full_data])

                    vib_list = np.array(vib_data, dtype=float)
                    x_list = np.array(x_data, dtype=float)
                    y_list = np.array(y_data, dtype=float)
                    z_list = np.array(z_data, dtype=float)
                    # total_momentum_list = np.array([float(d['total_momentum']) for d in full_data])
                    # total_velocity_list = np.array([float(d['total_velocity']) for d in full_data])

                    vib_filtered = self.bandpass_filter(vib_list)
                    x_filtered = self.bandpass_filter(x_list)
                    y_filtered = self.bandpass_filter(y_list)
                    z_filtered = self.bandpass_filter(z_list)
                    # total_momentum_filtered = self.bandpass_filter(total_momentum_list)
                    # total_velocity_filtered = self.bandpass_filter(total_velocity_list)
                    
                    # Ekstraksi fitur
                    feats = self.extract_features(vib_filtered, x_filtered, y_filtered, z_filtered)
                    print("[DEBUG] Extracted Features:\n", json.dumps(feats, indent=2, default=str))

                    # DataFrame dan scaler
                    df = pd.DataFrame([feats])[RoadMonitor.FEATURE_NAMES]
                    scaled_feats = self.scaler.transform(df)
                    pca_feats = self.pca.transform(scaled_feats)
                    print("[DEBUG] Scaled Features:\n", scaled_feats.tolist())

                    # Predicttt
                    pred = self.model.predict(pca_feats)[0]
                    print("[DEBUG] Raw Model Prediction:", pred)
                    
                    if hasattr(self.model, 'predict_proba'):
                        prob = self.model.predict_proba(pca_feats)
                        print("[DEBUG] Prediction Probabilities:", prob.tolist())
                    
                    ml_status = ['UNDULATING','NORMAL','SEVERE UNDULATION'][pred]
                    stats     = {
                        'max': np.max(vib_data),
                        'avg': np.mean(vib_data)
                    }
                except Exception as e:
                    print("ML error:", e)

                result = {
                    'chunk_number': chunk_count,
                    'status':       ml_status,
                    'location':     f"{gps['lat']:.5f}, {gps['lon']:.5f}",
                    'timestamp':    datetime.fromtimestamp(ntp_time, tz=timezone(timedelta(hours=7))).isoformat(),
                    'vibration_data': vib_data,
                    'stats':        stats,
                    'slope_deg':   slope_deg
                }

                # SSE
                self.last_result = result
                self.push_update(result)

                # → Webhook push
                try:
                    hook_payload = {'data': {
                        'lat': gps['lat'],
                        'lon': gps['lon'],
                        'alt': gps['alt'],
                        'status': ml_status,
                        'timestamp': result['timestamp'],
                        'vibration': vib_data,
                        'x': x_data,
                        'y': y_data,
                        'z': z_data,
                        'total_momentum_data': total_momentum_data,
                        'total_velocity_data': total_velocity_data
                    }}
                    # POST ke alpi

                    headers = {
                        'Authorization': 'Bearer ace19d66-b0b1-4e09-a313-853df1fd2c91'
                    }
                    requests.post(WEBHOOK_URL, json=hook_payload, timeout=5.0, headers=headers, verify=False)
                except Exception as e:
                    print("Webhook error:", e)

                time.sleep(10)
                chunk_count += 1

            except queue.Empty:
                continue
            except Exception as e:
                print("Processing loop error:", e)
                time.sleep(1)

monitor = RoadMonitor()

@app.route('/api/vibration', methods=['POST'])
def handle_vibration():
    try:
        data = request.get_json()
        with monitor.lock:
            monitor.raw_queue.put({
                'vibration': data['vibration'],
                'x': data['x'],
                'y': data['y'],
                'z': data['z'],
                'total_velocity': data['total_velocity'],
                'total_momentum': data['total_momentum'],
                'full_data': data['full_data'],
                'timestamp': data['timestamp']
            })
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
@app.route('/api/processed', methods=['GET'])
def get_processed():
    if monitor.last_result is None:
        return jsonify({'error': 'No data yet'}), 404

    res = monitor.last_result
    lat, lon = res['location'].split(', ')
    return jsonify({
        'data': {
            'lat':       lat,
            'lon':       lon,
            'status':    res['status'],
            'timestamp': res['timestamp'],
            'vibration': res['vibration_data'],
            'slope_deg': res.get('slope_deg', None),
        }
    })

@app.route('/stream')
def stream():
    def event_stream():
        client_queue = queue.Queue()
        client = {'queue': client_queue}
        monitor.clients.append(client)
        try:
            while True:
                data = client_queue.get()
                yield f"data: {json.dumps(data)}\n\n"
        finally:
            monitor.clients.remove(client)
    return Response(event_stream(), mimetype="text/event-stream")

@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
