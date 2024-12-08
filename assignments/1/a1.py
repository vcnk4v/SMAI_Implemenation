import argparse
import subprocess


def run_linreg():
    """Run the linear regression script."""
    subprocess.run(["python3", "a1_linreg.py"])


def run_knn():
    """Run the KNN script."""
    subprocess.run(["python3", "a1_knn.py"])


def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description="Run either Linear Regression or KNN script."
    )

    # Add argument to choose between linreg and knn
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["linreg", "knn"],
        help='Specify which model to run: "linreg" for Linear Regression, "knn" for KNN.',
    )

    # Parse the arguments
    args = parser.parse_args()

    # Run the appropriate script based on the argument
    if args.model == "linreg":
        run_linreg()
    elif args.model == "knn":
        run_knn()
    else:
        print("Invalid model selected. Please choose either 'linreg' or 'knn'.")


if __name__ == "__main__":
    main()
