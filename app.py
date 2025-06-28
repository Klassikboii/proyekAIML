from flask import Flask, render_template, request, jsonify
from matplotlib import pyplot as plt
from aco_schedule import ACO
import matplotlib
matplotlib.use('Agg')  # Untuk menghindari conflict dengan thread Flask
import os
from datetime import datetime

app = Flask(__name__)

# Konfigurasi direktori untuk menyimpan gambar
UPLOAD_FOLDER = 'static/results'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Data kota contoh
CITY_DATA = {
    "Jakarta": {"min_demand": 40, "priority": 1},
    "Bandung": {"min_demand": 30, "priority": 2},
    "Surabaya": {"min_demand": 35, "priority": 1},
    "Medan": {"min_demand": 25, "priority": 3},
    "Makassar": {"min_demand": 20, "priority": 3}
}

@app.route('/')
def index():
    """Halaman utama untuk input parameter"""
    return render_template('index.html', cities=CITY_DATA)

@app.route('/schedule', methods=['POST'])
def generate_schedule():
    """Endpoint untuk generate jadwal"""
    try:
        # 1. Ambil data dari form
        req_data = request.form
        
        # Parameter sistem
        total_capacity = float(req_data.get('total_capacity', 150))
        min_capacity = float(req_data.get('min_capacity', 100))
        critical_capacity = float(req_data.get('critical_capacity', 15))
        
        # Kebutuhan per kota
        city_demands = {}
        for city in CITY_DATA:
            demand = float(req_data.get(f'city_{city}', 0))
            if demand > 0:
                city_demands[city] = demand
        
        # Parameter ACO
        num_ants = int(req_data.get('num_ants', 14))
        num_iterations = int(req_data.get('num_iterations', 20))
        alpha = float(req_data.get('alpha', 1.0))
        beta = float(req_data.get('beta', 2.0))
        evaporation_rate = float(req_data.get('evaporation_rate', 0.3))
        
        # 2. Validasi input
        total_demand = sum(city_demands.values())
        if total_demand > total_capacity:
            return jsonify({
                'status': 'error',
                'message': f'Total kebutuhan kota ({total_demand} MW) melebihi kapasitas total ({total_capacity} MW)'
            })
        
        # 3. Inisialisasi scheduler
        scheduler = ACO(
            total_capacity=total_capacity,
            min_capacity=min_capacity,
            critical_capacity=critical_capacity,
            city_demands=city_demands
        )
        
        # 4. Jalankan ACO
        best_schedule, best_penalty = scheduler.run_aco(
            num_ants=num_ants,
            num_iterations=num_iterations,
            alpha=alpha,
            beta=beta,
            evaporation_rate=evaporation_rate,
            show_progress=False
        )
        
        # 5. Generate visualisasi
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        schedule_img = os.path.join(UPLOAD_FOLDER, f'schedule_{timestamp}.png')
        capacity_img = os.path.join(UPLOAD_FOLDER, f'capacity_{timestamp}.png')
        
        scheduler.visualize_schedule(best_schedule)
        plt.savefig(schedule_img)
        plt.close()
        
        scheduler.visualize_capacity(best_schedule)
        plt.savefig(capacity_img)
        plt.close()
        
        # 6. Format hasil
        schedule_data = []
        for node in best_schedule:
            schedule_data.append({
                'unit': node.unit,
                'capacity': node.capacity,
                'start': node.start,
                'end': node.end,
                'duration': node.end - node.start + 1
            })
        
        monthly_capacities = scheduler.calculate_monthly_capacities(best_schedule)
        
        # 7. Return response
        return jsonify({
            'status': 'success',
            'schedule': schedule_data,
            'monthly_capacities': monthly_capacities,
            'total_penalty': best_penalty,
            'total_demand': total_demand,
            'images': {
                'schedule': schedule_img,
                'capacity': capacity_img
            },
            'parameters': {
                'total_capacity': total_capacity,
                'min_capacity': min_capacity,
                'critical_capacity': critical_capacity
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
