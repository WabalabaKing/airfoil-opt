from airfoil_opt.active_optimizer import ActiveAirfoilOptimizer
from airfoil_opt.post_processing import run_all_post_processing

def main():
    optimizer = ActiveAirfoilOptimizer()
    optimizer.run()
    run_all_post_processing()

if __name__ == "__main__":
    main()