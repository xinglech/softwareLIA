# comparison_dashboard.py
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
import numpy as np

class LIAComparisonDashboard:
    def __init__(self, results_dir='lia_results'):
        self.results_dir = Path(results_dir)
        self.comparisons = []
    
    def load_all_results(self):
        """Load all available software and hardware results"""
        software_files = list(self.results_dir.glob('software_lia_*.json'))
        hardware_files = list(self.results_dir.glob('hardware_results_*.json'))
        comparison_files = list(self.results_dir.glob('comparison_*.json'))
        
        all_results = {
            'software': [],
            'hardware': [],
            'comparisons': []
        }
        
        for file in software_files:
            with open(file, 'r') as f:
                all_results['software'].append(json.load(f))
        
        for file in hardware_files:
            with open(file, 'r') as f:
                all_results['hardware'].append(json.load(f))
        
        for file in comparison_files:
            with open(file, 'r') as f:
                all_results['comparisons'].append(json.load(f))
        
        return all_results
    
    def create_summary_report(self):
        """Create a comprehensive summary report"""
        results = self.load_all_results()
        
        print("="*60)
        print("LIA COMPARISON DASHBOARD - SUMMARY REPORT")
        print("="*60)
        
        print(f"\nSOFTWARE LIA ANALYSES: {len(results['software'])}")
        for i, sw in enumerate(results['software']):
            meta = sw.get('metadata', {})
            acq = sw.get('acquisition_info', {})
            print(f"  {i+1}. {meta.get('timestamp', 'N/A')} - "
                  f"Source: {acq.get('data_source', 'N/A')} - "
                  f"R: {sw.get('R_mean', 0):.6f} V")
        
        print(f"\nHARDWARE LIA ANALYSES: {len(results['hardware'])}")
        for i, hw in enumerate(results['hardware']):
            stats = hw.get('summary_statistics', {})
            print(f"  {i+1}. {hw.get('timestamp', 'N/A')} - "
                  f"R: {stats.get('R_mean', 0):.6f} V")
        
        print(f"\nCOMPARISONS: {len(results['comparisons'])}")
        for i, comp in enumerate(results['comparisons']):
            diff = comp.get('differences', {})
            print(f"  {i+1}. {comp.get('timestamp', 'N/A')} - "
                  f"Amplitude Error: {diff.get('amplitude_relative_error', 0):.2f}%")
    
    def plot_trend_analysis(self):
        """Plot trends over multiple analyses"""
        results = self.load_all_results()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Software LIA trend
        if results['software']:
            sw_timestamps = [r.get('metadata', {}).get('timestamp', '') 
                           for r in results['software']]
            sw_amplitudes = [r.get('R_mean', 0) for r in results['software']]
            axes[0,0].plot(sw_timestamps, sw_amplitudes, 'bo-', label='Software LIA')
            axes[0,0].set_title('Software LIA - Amplitude Trend')
            axes[0,0].set_ylabel('Amplitude (V)')
            axes[0,0].tick_params(axis='x', rotation=45)
        
        # Hardware LIA trend  
        if results['hardware']:
            hw_timestamps = [r.get('timestamp', '') for r in results['hardware']]
            hw_amplitudes = [r.get('summary_statistics', {}).get('R_mean', 0) 
                           for r in results['hardware']]
            axes[0,1].plot(hw_timestamps, hw_amplitudes, 'ro-', label='Hardware LIA')
            axes[0,1].set_title('Hardware LIA - Amplitude Trend')
            axes[0,1].set_ylabel('Amplitude (V)')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # Comparison trend
        if results['comparisons']:
            comp_timestamps = [r.get('timestamp', '') for r in results['comparisons']]
            comp_errors = [r.get('differences', {}).get('amplitude_relative_error', 0) 
                         for r in results['comparisons']]
            axes[1,0].plot(comp_timestamps, comp_errors, 'go-', label='Error')
            axes[1,0].set_title('Amplitude Relative Error Trend')
            axes[1,0].set_ylabel('Error (%)')
            axes[1,0].tick_params(axis='x', rotation=45)
        
        axes[1,1].axis('off')
        axes[1,1].text(0.1, 0.9, "LIA COMPARISON DASHBOARD\n\n"
                      f"Total Analyses: {len(results['software']) + len(results['hardware'])}\n"
                      f"Comparisons: {len(results['comparisons'])}\n"
                      f"Data Directory: {self.results_dir}",
                      transform=axes[1,1].transAxes, fontsize=12)
        
        plt.tight_layout()
        plt.savefig('lia_trend_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    dashboard = LIAComparisonDashboard()
    dashboard.create_summary_report()
    dashboard.plot_trend_analysis()
