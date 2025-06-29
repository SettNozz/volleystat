#!/usr/bin/env python3
"""
Quick Training Progress Viewer
Simple utility to view the latest training metrics and plots.
"""

import os
import sys
from pathlib import Path
import subprocess


def find_latest_training_run(models_dir: str) -> Path:
    """Find the most recent training run directory."""
    models_path = Path(models_dir)
    
    if not models_path.exists():
        return None
    
    # Look for YOLO training directories
    training_dirs = []
    for run_dir in models_path.iterdir():
        if run_dir.is_dir() and ('yolov8' in run_dir.name and 'volleyball' in run_dir.name):
            training_dirs.append(run_dir)
    
    if not training_dirs:
        return None
    
    # Return the most recently modified
    return max(training_dirs, key=lambda x: x.stat().st_mtime)


def show_training_info(training_dir: Path) -> None:
    """Display training information and available files."""
    print(f"🏃 Training Run: {training_dir.name}")
    print(f"📁 Location: {training_dir}")
    print("=" * 60)
    
    # Check for results.csv
    results_csv = training_dir / "results.csv"
    if results_csv.exists():
        print(f"✅ Results CSV: {results_csv}")
        
        # Quick stats from CSV
        try:
            import pandas as pd
            df = pd.read_csv(results_csv)
            print(f"📊 Training Progress: {len(df)} epochs completed")
            
            if 'metrics/mAP50(B)' in df.columns:
                latest_map = df['metrics/mAP50(B)'].iloc[-1]
                best_map = df['metrics/mAP50(B)'].max()
                best_epoch = df.loc[df['metrics/mAP50(B)'].idxmax(), 'epoch']
                print(f"🎯 Latest mAP@0.5: {latest_map:.4f}")
                print(f"🏆 Best mAP@0.5: {best_map:.4f} (epoch {best_epoch})")
            
        except Exception as e:
            print(f"⚠️  Could not read results: {e}")
    else:
        print("❌ No results.csv found")
    
    # Check for plots
    plots_dir = training_dir / "metrics_plots"
    if plots_dir.exists():
        plot_files = list(plots_dir.glob("*.png"))
        print(f"📊 Metrics plots: {len(plot_files)} files in {plots_dir}")
        
        # Find latest plot
        latest_plot = plots_dir / "latest_metrics.png"
        if latest_plot.exists():
            print(f"📈 Latest plot: {latest_plot}")
        
        # List recent plots
        timestamped_plots = [p for p in plot_files if "epoch_" in p.name]
        if timestamped_plots:
            recent_plots = sorted(timestamped_plots, key=lambda x: x.stat().st_mtime, reverse=True)[:3]
            print("📋 Recent plots:")
            for plot in recent_plots:
                print(f"   - {plot.name}")
    else:
        print("❌ No metrics plots directory found")
    
    # Check for checkpoints
    weights_dir = training_dir / "weights"
    if weights_dir.exists():
        checkpoints = list(weights_dir.glob("*.pt"))
        print(f"💾 Checkpoints: {len(checkpoints)} files")
        
        for checkpoint in ['best.pt', 'last.pt']:
            checkpoint_path = weights_dir / checkpoint
            if checkpoint_path.exists():
                size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
                print(f"   ✅ {checkpoint}: {size_mb:.1f} MB")
    else:
        print("❌ No weights directory found")


def open_plot(plot_path: Path) -> None:
    """Open plot file with default image viewer."""
    try:
        if os.name == 'nt':  # Windows
            os.startfile(plot_path)
        elif os.name == 'posix':  # macOS and Linux
            subprocess.run(['open' if sys.platform == 'darwin' else 'xdg-open', str(plot_path)])
        
        print(f"📊 Opened plot: {plot_path.name}")
    except Exception as e:
        print(f"❌ Could not open plot: {e}")
        print(f"📁 Manual path: {plot_path}")


def main():
    """Main function."""
    print("👁️  Training Progress Viewer")
    print("=" * 40)
    
    # Set models directory
    models_dir = r"C:\Users\illya\Documents\volleyball_analitics\volleystat\models\yolov8_curated_training"
    
    # Find latest training run
    latest_run = find_latest_training_run(models_dir)
    
    if not latest_run:
        print(f"❌ No training runs found in {models_dir}")
        print("\n💡 Make sure you have run training at least once:")
        print("   python run_curated_training.py")
        return
    
    # Show training info
    show_training_info(latest_run)
    
    # Interactive options
    print("\n🔧 Available Actions:")
    print("  1. View latest metrics plot")
    print("  2. Create new comprehensive plot")
    print("  3. Open training directory")
    print("  4. Show detailed metrics")
    print("  5. Exit")
    
    while True:
        try:
            choice = input("\nSelect action (1-5): ").strip()
            
            if choice == '1':
                latest_plot = latest_run / "metrics_plots" / "latest_metrics.png"
                if latest_plot.exists():
                    open_plot(latest_plot)
                else:
                    print("❌ Latest plot not found")
                    
            elif choice == '2':
                results_csv = latest_run / "results.csv"
                if results_csv.exists():
                    print("📊 Creating comprehensive plot...")
                    try:
                        # Import and use the plotter
                        script_dir = Path(__file__).parent
                        sys.path.append(str(script_dir))
                        from plot_training_metrics import MetricsPlotter
                        
                        plotter = MetricsPlotter(str(results_csv))
                        plotter.create_comprehensive_plot()
                        print("✅ Plot created successfully!")
                    except Exception as e:
                        print(f"❌ Error creating plot: {e}")
                else:
                    print("❌ No results.csv found")
                    
            elif choice == '3':
                try:
                    if os.name == 'nt':  # Windows
                        os.startfile(latest_run)
                    print(f"📁 Opened: {latest_run}")
                except Exception as e:
                    print(f"❌ Could not open directory: {e}")
                    print(f"📁 Manual path: {latest_run}")
                    
            elif choice == '4':
                results_csv = latest_run / "results.csv"
                if results_csv.exists():
                    try:
                        import pandas as pd
                        df = pd.read_csv(results_csv)
                        
                        print("\n📊 Detailed Training Metrics:")
                        print("-" * 40)
                        print(f"Total epochs: {len(df)}")
                        
                        # Show progression every 10 epochs
                        milestone_epochs = range(10, len(df) + 1, 10)
                        
                        if 'metrics/mAP50(B)' in df.columns:
                            print("\nmAP@0.5 progression (every 10 epochs):")
                            for epoch in milestone_epochs:
                                if epoch <= len(df):
                                    map_val = df.iloc[epoch-1]['metrics/mAP50(B)']
                                    print(f"  Epoch {epoch:3d}: {map_val:.4f}")
                                    
                    except Exception as e:
                        print(f"❌ Error reading metrics: {e}")
                else:
                    print("❌ No results.csv found")
                    
            elif choice == '5':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-5")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main() 