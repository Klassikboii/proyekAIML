import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple

class Node:
    def __init__(self, unit: int, capacity: float, start_month: int, end_month: int):
        """
        Representasi jadwal maintenance untuk satu unit pembangkit
        
        Parameters:
            unit (int): Nomor unit pembangkit (1-7)
            capacity (float): Kapasitas yang tidak tersedia saat maintenance (MW)
            start_month (int): Bulan mulai maintenance (1-12)
            end_month (int): Bulan selesai maintenance (1-12)
        """
        self.unit = unit
        self.capacity = capacity
        self.start = start_month
        self.end = end_month
    
    def __str__(self):
        return f"Unit {self.unit}: Bulan {self.start}-{self.end} (Kapasitas: {self.capacity}MW)"

class PowerPlantACOScheduler:
    def __init__(self, total_capacity: float = 150, min_capacity: float = 100, 
                 critical_capacity: float = 15, city_demands: dict = None):
        """
        Inisialisasi scheduler dengan parameter sistem
        
        Parameters:
            total_capacity (float): Kapasitas total sistem (MW)
            min_capacity (float): Kapasitas minimum yang harus tersedia (MW)
            critical_capacity (float): Kapasitas kritis sistem (MW)
            city_demands (dict): Kebutuhan listrik per kota {'city_name': demand}
        """
        self.total_capacity = total_capacity
        self.min_capacity = min_capacity
        self.critical_capacity = critical_capacity
        self.city_demands = city_demands or {}
        
        # Spesifikasi unit pembangkit (unit: {capacity, interval})
        self.unit_specs = {
            1: {'capacity': 20, 'interval': 2},
            2: {'capacity': 15, 'interval': 2},
            3: {'capacity': 35, 'interval': 1},
            4: {'capacity': 40, 'interval': 1},
            5: {'capacity': 15, 'interval': 1},
            6: {'capacity': 15, 'interval': 2},
            7: {'capacity': 10, 'interval': 1}
        }
        
        # Generate semua kemungkinan jadwal maintenance
        self.nodes = self._initialize_nodes()
        self.num_units = len(self.unit_specs)
    
    def _initialize_nodes(self) -> List[Node]:
        """Generate semua kemungkinan jadwal maintenance untuk semua unit"""
        nodes = []
        for unit, specs in self.unit_specs.items():
            for month in range(1, 13):
                start = month
                end = month + specs['interval'] - 1
                if end <= 12:  # Hanya sampai bulan 12
                    nodes.append(Node(unit, specs['capacity'], start, end))
        return nodes
    
    def is_valid_addition(self, current_solution: List[Node], candidate_node: Node) -> bool:
        """
        Cek apakah node kandidat bisa ditambahkan ke solusi saat ini
        
        Rules:
        1. Setiap unit maksimal 2 jadwal maintenance
        2. Tidak ada overlap jadwal untuk unit yang sama
        3. Untuk unit yang sudah ada 1 jadwal, jadwal kedua harus tidak overlap
        """
        unit_counts = [0] * self.num_units
        unit_schedules = {unit: [] for unit in self.unit_specs}
        
        # Hitung jadwal per unit dan kumpulkan periodenya
        for node in current_solution:
            unit_counts[node.unit - 1] += 1
            unit_schedules[node.unit - 1].append((node.start, node.end))
        
        # Rule 1: Maksimal 2 jadwal per unit
        if unit_counts[candidate_node.unit - 1] >= 2:
            return False
        
        # Rule 2 & 3: Cek overlap untuk unit yang sama
        for start, end in unit_schedules[candidate_node.unit - 1]:
            if not (end < candidate_node.start or start > candidate_node.end):
                return False  # Overlap ditemukan
        
        return True
    
    def calculate_penalty(self, schedule: List[Node]) -> float:
        """
        Hitung total penalty untuk sebuah solusi/jadwal
        
        Penalty dihitung berdasarkan:
        1. Kekurangan kapasitas dari kebutuhan kota
        2. Pelanggaran kapasitas minimum sistem
        3. Pelanggaran kapasitas kritis
        """
        monthly_capacities = np.full(12, self.total_capacity)
        
        # Kurangi kapasitas untuk bulan-bulan maintenance
        for node in schedule:
            monthly_capacities[node.start-1:node.end] -= node.capacity
        
        total_penalty = 0
        
        # 1. Penalty kebutuhan kota
        if self.city_demands:
            total_demand = sum(self.city_demands.values())
            for month in range(12):
                if monthly_capacities[month] < total_demand:
                    total_penalty += (total_demand - monthly_capacities[month]) * 2  # Bobot lebih besar
        
        # 2. Penalty kapasitas minimum
        for capacity in monthly_capacities:
            if capacity < self.min_capacity:
                total_penalty += (self.min_capacity - capacity) * 1.5
        
        # 3. Penalty kapasitas kritis
        for capacity in monthly_capacities:
            if capacity < self.critical_capacity:
                total_penalty += (self.critical_capacity - capacity) * 3  # Bobot paling besar
        
        return max(total_penalty, 0.1)  # Minimal 0.1 untuk hindari division by zero
    
    def run_aco(self, num_ants: int = 14, num_iterations: int = 50, 
                alpha: float = 1, beta: float = 2, evaporation_rate: float = 0.3,
                show_progress: bool = True) -> Tuple[List[Node], float]:
        """
        Jalankan algoritma ACO untuk mencari jadwal optimal
        
        Parameters:
            num_ants (int): Jumlah semut (solusi) per iterasi
            num_iterations (int): Jumlah iterasi
            alpha (float): Parameter pengaruh pheromone
            beta (float): Parameter pengaruh heuristik
            evaporation_rate (float): Tingkat penguapan pheromone
            show_progress (bool): Tampilkan progress tiap iterasi
            
        Returns:
            Tuple[List[Node], float]: Solusi terbaik dan nilai fitness-nya
        """
        # Inisialisasi matriks pheromone (unit x unit)
        pheromone = np.ones((self.num_units, self.num_units))
        
        best_solution = None
        best_fitness = 0  # Fitness = 1/penalty
        history = []
        
        for iteration in range(num_iterations):
            solutions = []
            
            # Setiap semut membangun solusi
            for _ in range(num_ants):
                solution = self._construct_solution(pheromone, alpha, beta)
                solutions.append(solution)
            
            # Evaluasi semua solusi
            for solution in solutions:
                penalty = self.calculate_penalty(solution)
                fitness = 1 / penalty
                
                # Update solusi terbaik
                if fitness > best_fitness:
                    best_solution = solution
                    best_fitness = fitness
                elif fitness == best_fitness and self._is_different_solution(solution, [best_solution]):
                    best_solution = solution  # Pertahankan diversitas
            
            # Update pheromone
            pheromone = self._update_pheromone(pheromone, solutions, evaporation_rate)
            
            # Catat history untuk visualisasi
            history.append(1/best_fitness if best_fitness > 0 else float('inf'))
            
            # Tampilkan progress
            if show_progress:
                print(f"Iterasi {iteration+1}: Best Penalty = {history[-1]:.2f}")
        
        # Visualisasi perkembangan penalty
        self._plot_convergence(history)
        
        return best_solution, 1/best_fitness if best_fitness > 0 else float('inf')
    
    def _construct_solution(self, pheromone: np.ndarray, alpha: float, beta: float) -> List[Node]:
        """Bangun solusi dengan aturan probabilistik ACO"""
        solution = []
        
        # Mulai dengan node random
        solution.append(random.choice(self.nodes))
        
        # Bangun solusi sampai semua unit memiliki 2 jadwal
        while len(solution) < 2 * self.num_units:
            candidates = []
            probabilities = []
            last_unit = solution[-1].unit
            
            for node in self.nodes:
                if self.is_valid_addition(solution, node):
                    candidates.append(node)
                    
                    # Hitung nilai heuristik
                    current_penalty = self.calculate_penalty(solution)
                    new_penalty = self.calculate_penalty(solution + [node])
                    heuristic = 1 / (new_penalty - current_penalty + 1e-6)  # Hindari division by zero
                    
                    # Hitung probabilitas
                    p = (pheromone[last_unit-1][node.unit-1] ** alpha) * (heuristic ** beta)
                    probabilities.append(p)
            
            # Normalisasi probabilitas
            if probabilities:
                prob_sum = sum(probabilities)
                if prob_sum > 0:
                    probabilities = [p/prob_sum for p in probabilities]
                    selected_node = random.choices(candidates, weights=probabilities)[0]
                    solution.append(selected_node)
                else:
                    # Jika semua probabilitas 0, pilih random
                    selected_node = random.choice(candidates)
                    solution.append(selected_node)
            else:
                break  # Tidak ada kandidat valid
        
        return solution
    
    def _update_pheromone(self, pheromone: np.ndarray, solutions: List[List[Node]], 
                         evaporation_rate: float) -> np.ndarray:
        """Update pheromone dengan evaporation dan deposit"""
        # Evaporation
        pheromone *= (1 - evaporation_rate)
        
        # Deposit pheromone untuk semua solusi
        for solution in solutions:
            fitness = 1 / self.calculate_penalty(solution)
            
            # Deposit pheromone pada path yang dilalui
            for i in range(len(solution)-1):
                from_unit = solution[i].unit - 1
                to_unit = solution[i+1].unit - 1
                pheromone[from_unit][to_unit] += fitness
        
        return pheromone
    
    def _is_different_solution(self, solution1: List[Node], solutions: List[List[Node]]) -> bool:
        """Cek apakah solusi berbeda dengan yang sudah ada"""
        set1 = {(node.unit, node.start, node.end) for node in solution1}
        for sol in solutions:
            set2 = {(node.unit, node.start, node.end) for node in sol}
            if set1 == set2:
                return False
        return True
    
    def _plot_convergence(self, history: List[float]):
        """Visualisasi perkembangan penalty selama iterasi"""
        plt.figure(figsize=(10, 5))
        plt.plot(history, marker='o', linestyle='-', color='b')
        plt.title("Konvergensi Algoritma ACO")
        plt.xlabel("Iterasi")
        plt.ylabel("Penalty Terbaik")
        plt.grid(True)
        plt.show()
    
    def visualize_schedule(self, schedule: List[Node]):
        """Visualisasi jadwal maintenance dalam bentuk Gantt chart"""
        plt.figure(figsize=(12, 6))
        
        for node in schedule:
            plt.barh(
                y=f"Unit {node.unit}",
                width=node.end - node.start + 1,
                left=node.start,
                color=f'C{node.unit-1}',
                edgecolor='black',
                label=f"Unit {node.unit}" if node.start == 1 else ""
            )
        
        plt.title("Jadwal Maintenance Pembangkit Listrik")
        plt.xlabel("Bulan")
        plt.ylabel("Unit Pembangkit")
        plt.xticks(range(1, 13))
        plt.yticks(range(1, self.num_units+1))
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def visualize_capacity(self, schedule: List[Node]):
        """Visualisasi kapasitas tersedia per bulan"""
        months = range(1, 13)
        capacities = np.full(12, self.total_capacity)
        
        for node in schedule:
            capacities[node.start-1:node.end] -= node.capacity
        
        plt.figure(figsize=(10, 5))
        plt.plot(months, capacities, marker='o', linestyle='-', color='b', label='Kapasitas Tersedia')
        
        # Tambahkan garis kebutuhan jika ada data kota
        if self.city_demands:
            total_demand = sum(self.city_demands.values())
            plt.axhline(y=total_demand, color='r', linestyle='--', label='Total Kebutuhan Kota')
        
        plt.axhline(y=self.min_capacity, color='orange', linestyle=':', label='Kapasitas Minimum')
        plt.axhline(y=self.critical_capacity, color='purple', linestyle=':', label='Kapasitas Kritis')
        
        plt.title("Kapasitas Pembangkit Listrik per Bulan")
        plt.xlabel("Bulan")
        plt.ylabel("Kapasitas (MW)")
        plt.xticks(months)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

# Contoh Penggunaan
if __name__ == "__main__":
    # Data kebutuhan kota
    city_demands = {
        "Jakarta": 40,
        "Bandung": 30,
        "Surabaya": 35,
        "Medan": 25,
        "Makassar": 20
    }
    
    # Inisialisasi scheduler
    scheduler = PowerPlantACOScheduler(
        total_capacity=150,
        min_capacity=100,
        critical_capacity=15,
        city_demands=city_demands
    )
    
    # Jalankan ACO
    best_schedule, best_penalty = scheduler.run_aco(
        num_ants=14,
        num_iterations=20,
        alpha=1,
        beta=2,
        evaporation_rate=0.3
    )
    
    # Tampilkan hasil
    print("\n=== Solusi Terbaik ===")
    print(f"Total Penalty: {best_penalty:.2f}")
    for node in best_schedule:
        print(node)
    
    # Visualisasi
    scheduler.visualize_schedule(best_schedule)
    scheduler.visualize_capacity(best_schedule)
