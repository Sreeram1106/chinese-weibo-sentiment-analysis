"""One command to reproduce everything:  python run_all.py   (≈3 min)"""
from src import inspect_dataset, leakage_audit, train

if __name__ == "__main__":
    print("== 1/3 dataset audit ==");   inspect_dataset.main()
    print("\n== 2/3 train + evaluate =="); train.main()
    print("\n== 3/3 leakage audit ==");   leakage_audit.main()
    print("\nDone. Launch the demo with:  streamlit run app.py")
