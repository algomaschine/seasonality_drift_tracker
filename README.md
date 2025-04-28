# Enhanced Cycle Scanner

**Author:** Eduard Samokhvalov

## Description
The Enhanced Cycle Scanner is an advanced open-source tool for detecting, validating, and forecasting cycles in financial time series, with a focus on Bitcoin price data. It implements robust trend and seasonality decomposition, multi-timeframe and multi-variant analysis, and generates interactive reports. The code is released under the GNU General Public License v3.0 (GPLv3) and is specifically granted for use by the Foundation for the Study of Cycles.

## Key Features
- **Multi-Timeframe Analysis:** Analyze cycles across a Fibonacci progression of timeframes from 1 hour to 1 month.
- **Major High/Low Segmentation:** For each timeframe, analyzes the latest segment from the most recent major high or low, with configurable segment lengths.
- **Advanced Decomposition:** Handles trend, evolving seasonality (drift), and residuals using state space models and dynamic harmonic regression.
- **Cycle Validation:** Uses FFT and the Bartels test for robust cycle detection and statistical validation.
- **Interactive Reporting:** Generates HTML reports for each variant and an "uber-report" that ranks and links all results.
- **Parallel Processing:** Efficient multiprocessing for fast analysis across all variants.
- **Open Source:** Licensed under GPLv3, with explicit grant for the Foundation for the Study of Cycles.

## Installation

### Requirements
- Python 3.8+
- pip (Python package manager)

### Dependencies
- pandas
- numpy
- scipy
- statsmodels
- plotly
- fpdf

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/algomaschine
   cd enhanced-cycle-scanner
   ```
2. **Run the main scanner:**
   ```bash
   python enhanced_cycle_scanner.py
   ```
   - Reports will be saved in the `reports/` directory, organized by timeframe and variant.
   - The `uber_report.html` provides a ranked overview and links to all variant reports.

3. **Generate the white paper (PDF):**
   ```bash
   python reports/cycle_scanner_whitepaper.py
   ```
   - The PDF will be saved as `cycle_scanner_whitepaper.pdf`.

## License
This project is licensed under the GNU General Public License v3.0 (GPLv3).

- Copyright 2025 Eduard Samokhvalov
- Additional Grant: This software is specifically granted for use by the "Foundation for the Study of Cycles".
- See the LICENSE file for the full license text.

## Code Availability
The code is open source and available at: https://github.com/algomaschine

## Contributing
Contributions are welcome! Please open issues or submit pull requests.

## Contact
For questions or collaboration, contact Eduard Samokhvalov at: edward.samokhvalov@gmail.com 
