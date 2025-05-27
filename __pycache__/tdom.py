import pandas as pd
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np  # Import numpy
from datetime import datetime, timedelta
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel,
    QComboBox, QPushButton, QMessageBox, QTextEdit,
    QSpinBox, QDateEdit, QListWidget, QAbstractItemView,
    QHBoxLayout)  # Added QListWidget and QHBoxLayout
from PyQt5.QtCore import QDate

class TDOMAnalyzerApp(QWidget):
    HOLIDAYS = [
        datetime(2025, 2, 17),  # Presidents Day
        datetime(2025, 4, 18),  # Good Friday
        datetime(2025, 5, 26),  # Memorial Day
        datetime(2025, 6, 19),  # Juneteenth
        datetime(2025, 7, 4),   # Independence Day
        datetime(2025, 9, 1),   # Labor Day
        datetime(2025, 11, 27), # Thanksgiving
        datetime(2025, 12, 25)  # Christmas
    ]

    def __init__(self):
        super().__init__()
        self.data = None
        self.selected_symbols = []  # To store selected symbols
        self.initUI()
        self.load_initial_data()

    def load_initial_data(self):
try:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    # Load the parquet file instead of CSV
    data = pd.read_parquet(file_path)
    data.columns = [col.lower().strip() for col in data.columns]
    required_columns = {'date', 'symbol', 'tdom', 'profit'}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data = data.dropna(subset=['date'])
    data['symbol'] = data['symbol'].str.strip().str.upper()
    if len(data) == 0:
        raise ValueError("No valid data rows found after processing")
    return data
except Exception as e:
    self.show_error_message(f"Error loading data: {str(e)}")
    return None

    def show_error_message(self, message):
        QMessageBox.critical(self, 'Error', message)

    def fetch_data_from_file(self, file_path):
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            data = pd.read_csv(file_path, delimiter=',', encoding='utf-8', on_bad_lines='skip')
            data.columns = [col.lower().strip() for col in data.columns]
            required_columns = {'date', 'symbol', 'tdom', 'profit'}
            missing_columns = required_columns - set(data.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            data['date'] = pd.to_datetime(data['date'], errors='coerce')
            data = data.dropna(subset=['date'])
            data['symbol'] = data['symbol'].str.strip().str.upper()
            if len(data) == 0:
                raise ValueError("No valid data rows found after processing")
            return data
        except Exception as e:
            self.show_error_message(f"Error loading data: {str(e)}")
            return None

    def calculate_z_scores(self, data):
        # More efficient z-score calculation using vectorized operations
        grouped = data.groupby('symbol')
        result_df = data.copy()
        result_df['z'] = (
            (data['profit'] - grouped['profit'].transform('mean')) /
            grouped['profit'].transform('std').replace(0, 1)  # Avoid division by zero
        )
        return result_df

    def initUI(self):
        self.setWindowTitle('TDOM Analyzer')
        self.setGeometry(100, 100, 400, 600)  # Adjusted height
        layout = QVBoxLayout()
        font = self.font()
        font.setPointSize(12)
        self.setFont(font)

        # Date selection
        self.date_label = QLabel('Select Date:')
        self.date_input = QDateEdit(self)
        self.date_input.setCalendarPopup(True)
        self.date_input.setDate(QDate.currentDate())
        self.date_input.dateChanged.connect(self.update_tdom)
        layout.addWidget(self.date_label)
        layout.addWidget(self.date_input)

        # Symbol selection
        self.symbol_label = QLabel('Select Symbol:')
        self.symbol_input = QComboBox(self)

        # Symbol selection layout
        symbol_layout = QHBoxLayout()
        symbol_layout.addWidget(self.symbol_label)
        symbol_layout.addWidget(self.symbol_input)

        # Select Symbol Button
        self.select_symbol_button = QPushButton("Select", self)
        self.select_symbol_button.clicked.connect(self.add_selected_symbol)
        symbol_layout.addWidget(self.select_symbol_button)
        layout.addLayout(symbol_layout)

        # Selected Symbols Display
        self.selected_symbols_label = QLabel("Selected Symbols:")
        layout.addWidget(self.selected_symbols_label)
        self.selected_symbols_list = QListWidget(self)
        layout.addWidget(self.selected_symbols_list)
        # Remove Symbol Button
        self.remove_symbol_button = QPushButton("Remove", self)
        self.remove_symbol_button.clicked.connect(self.remove_selected_symbol)
        layout.addWidget(self.remove_symbol_button)

        # TDOM input
        self.tdom_label = QLabel('Trading Day of Month (TDOM):')
        self.tdom_input = QSpinBox(self)
        self.tdom_input.setMinimum(1)
        self.tdom_input.setMaximum(30)
        layout.addWidget(self.tdom_label)
        layout.addWidget(self.tdom_input)

        # Weekday selection
        self.weekday_label = QLabel('Select Weekday:')
        self.weekday_input = QComboBox(self) # ##FIXED: change to combo box from list
        self.weekday_input.addItems(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']) # ##FIXED: change to combo box from list
        layout.addWidget(self.weekday_label) # ##FIXED: change to combo box from list
        layout.addWidget(self.weekday_input) # ##FIXED: change to combo box from list

        # Analyze button
        self.analyze_button = QPushButton('Analyze', self)
        self.analyze_button.clicked.connect(self.analyze_data)
        layout.addWidget(self.analyze_button)

        # Results display
        self.result_text = QTextEdit(self)
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)

        self.setLayout(layout)
        self.setMinimumSize(400, 450)
        self.load_symbols()
        self.update_tdom()

    def load_symbols(self):
        # Define market segments
        energy = {
            'CL': 'Crude Oil (WTI)',
            'NG': 'Natural Gas',
            'HO': 'Heating Oil',
            'RB': 'RBOB Gasoline',
            'BZ': 'Brent Crude Oil'
        }

        metals = {
            'GC': 'Gold',
            'SI': 'Silver',
            'HG': 'Copper',
            'PL': 'Platinum',
            'PA': 'Palladium'
        }

        agriculture = {
            'ZS': 'Soybeans',
            'ZM': 'Soybean Meal',
            'ZL': 'Soybean Oil',
            'ZC': 'Corn',
            'ZW': 'Wheat',
            'KE': 'KC HRW Wheat',
            'ZO': 'Oats',
            'HE': 'Lean Hogs',
            'LE': 'Live Cattle',
            'GF': 'Feeder Cattle',
            'DC': 'Class III Milk',
            'SB': 'Sugar',
            'KC': 'Coffee',
            'CT': 'Cotton',
            'CC': 'Cocoa',
            'LB': 'Lumber'
        }

        treasuries = {
            'ZN': '10-Year U.S. Treasury Note',
            'ZT': '2-Year U.S. Treasury Note',
            'ZF': '5-Year U.S. Treasury Note',
            'ZB': '30-Year U.S. Treasury Bond',
            'UB': 'Ultra U.S. Treasury Bond',
            'GE': 'Eurodollar'
        }

        indices = {
            'ES': 'S&P 500 E-mini',
            'NQ': 'Nasdaq 100 E-mini',
            'RTY': 'Russell 2000 E-mini'
        }

        forex = {
            '6E': 'Euro FX',
            '6J': 'Japanese Yen',
            '6B': 'British Pound',
            '6A': 'Australian Dollar',
            '6C': 'Canadian Dollar',
            '6S': 'Swiss Franc',
            '6N': 'New Zealand Dollar'
        }

        cryptocurrencies = {
            'BTC': 'Bitcoin Futures',
            'ETH': 'Ethereum Futures'
        }

        # Add segment headers and symbols to the combo box
        segments = [
            ("Energy", energy),
            ("Metals", metals),
            ("Ag", agriculture),
            ("Treasuries", treasuries),
            ("Indices", indices),
            ("Forex", forex),
            ("Crypto", cryptocurrencies)
        ]

        for segment_name, segment_dict in segments:
            # Add segment header as a non-selectable item
            self.symbol_input.addItem(f"--- {segment_name} ---")
            current_index = self.symbol_input.count() - 1
            self.symbol_input.model().item(current_index).setEnabled(False)
            
            # Add symbols for this segment
            for symbol, name in segment_dict.items():
                self.symbol_input.addItem(f"{name} ({symbol})", symbol)

    def calculate_tdom(self, selected_date):
        first_day = selected_date.replace(day=1)
        tdom = 1
        current_day = first_day
        while current_day < selected_date:
            if current_day.weekday() < 5 and current_day not in self.HOLIDAYS:
                tdom += 1
            current_day += timedelta(days=1)
        return tdom

    def update_tdom(self):
        selected_date = self.date_input.date().toPyDate()
        calculated_tdom = self.calculate_tdom(selected_date)
        self.tdom_input.setValue(calculated_tdom)
        self.weekday_input.setCurrentText(selected_date.strftime('%A'))

    def add_selected_symbol(self):
        symbol = self.symbol_input.currentText()
        if symbol not in [self.selected_symbols_list.item(i).text() for i in range(self.selected_symbols_list.count())]:
            self.selected_symbols_list.addItem(symbol)

    def remove_selected_symbol(self):
        selected_items = self.selected_symbols_list.selectedItems()
        for item in selected_items:
            self.selected_symbols_list.takeItem(self.selected_symbols_list.row(item))

    def analyze_data(self):
        selected_symbols = [
            item.text().split('(')[-1].strip(')')
            for item in [self.selected_symbols_list.item(i) 
                        for i in range(self.selected_symbols_list.count())]
        ]
        
        if not selected_symbols:
            QMessageBox.warning(self, 'Input Error', 'Please select at least one symbol.')
            return

        tdom_value = self.tdom_input.value()
        selected_weekday = self.weekday_input.currentText()
        selected_date = self.date_input.date().toPyDate()
        selected_month = selected_date.month

        # Filter data once for all symbols
        symbol_data = self.data[self.data['symbol'].isin(selected_symbols)]
        all_results = []

        for symbol in selected_symbols:
            current_symbol_data = symbol_data[symbol_data['symbol'] == symbol]
            if current_symbol_data.empty:
                all_results.append(f"No data available for symbol: {symbol}\n\n")
                continue

            # Calculate all cases at once using boolean masks
            base_mask = current_symbol_data['tdom'] == tdom_value
            weekday_mask = current_symbol_data['date'].dt.day_name() == selected_weekday
            month_mask = current_symbol_data['date'].dt.month == selected_month

            cases = [
                current_symbol_data[base_mask],  # Case 1
                current_symbol_data[base_mask & weekday_mask],  # Case 2
                current_symbol_data[base_mask & month_mask],  # Case 3
                current_symbol_data[base_mask & weekday_mask & month_mask]  # Case 4
            ]

            metrics = [
                (
                    round(case['profit'].mean()) if not case.empty else 0,
                    self.calculate_win_rate(case),
                    len(case)
                ) for case in cases
            ]

            result = self.format_analysis_results(symbol, tdom_value, selected_weekday, 
                                           selected_date, metrics)
            all_results.append(result)

        self.result_text.setPlainText(''.join(all_results))
        self.create_heatmap(selected_date, symbol_data, selected_symbols)

    def calculate_win_rate(self, data):
        if data.empty:
            return 0
        total_trades = len(data)
        winning_trades = len(data[data['profit'] > 0])
        return round((winning_trades / total_trades) * 100) if total_trades > 0 else 0

    def format_analysis_results(self, symbol, tdom_value, selected_weekday, selected_date, metrics):
        result = f"\nAnalysis for {symbol}:\n"
        result += f"TDOM: {tdom_value}\n"
        result += f"Date: {selected_date.strftime('%Y-%m-%d')}\n"
        result += f"Weekday: {selected_weekday}\n\n"

        cases = [
            f"All TDOM {tdom_value}",
            f"All TDOM {tdom_value} on {selected_weekday}s",
            f"TDOM {tdom_value} in {selected_date.strftime('%B')}",
            f"TDOM {tdom_value} on {selected_weekday}s in {selected_date.strftime('%B')}"
        ]

        for case, (avg_profit, win_rate, sample_size) in zip(cases, metrics):
            result += f"{case}:\n"
            result += f"  Average Profit: ${avg_profit:,}\n"
            result += f"  Win Rate: {win_rate}%\n"
            result += f"  Sample Size: {sample_size}\n\n"

        return result

    def create_heatmap(self, selected_date, symbol_data, selected_symbols):
        selected_month = selected_date.month

        if symbol_data.empty:
            QMessageBox.warning(self, 'Data Error', "No data available for any of the selected symbols.")
            return

        # Filter data more efficiently
        filtered_data = symbol_data[
            (symbol_data['date'].dt.month == selected_month) &
            (symbol_data['tdom'].between(1, 21))
        ]

        # Create pivot table directly instead of multiple operations
        heatmap_data = filtered_data.pivot_table(
            values='z',
            index='tdom',
            columns='symbol',
            aggfunc='mean'
        ).reindex(range(1, 22))

        # Ensure all selected symbols are present
        missing_symbols = set(selected_symbols) - set(heatmap_data.columns)
        for symbol in missing_symbols:
            heatmap_data[symbol] = np.nan

        self.plot_heatmap(heatmap_data[selected_symbols], selected_symbols)

    def plot_heatmap(self, heatmap_data, selected_symbols):
        plt.figure(figsize=(len(selected_symbols) * 2, 8))  # Adjust size based on the number of symbols

        # explicitly convert NaN values to 0 before plotting
        heatmap_data_filled = heatmap_data[selected_symbols].fillna(0)

        ax = sns.heatmap(
            heatmap_data_filled,  # Use only selected symbols and filled data
            cmap='RdYlGn',
            center=0,
            annot=True,
            fmt=".2f",
            cbar=False,  # Remove the legend
            linewidths=0,  # Remove thin black borders
        )


        plt.ylabel('TDOM', fontsize=12)  # Vertical axis label

        # Turn off the x-axis label (symbols)
      
        ax.set_xlabel('')  # This ensures no label is displayed for the x-axis


        selected_tdom = self.tdom_input.value()
        if selected_tdom in heatmap_data.index:
            tdom_idx = heatmap_data.index.get_loc(selected_tdom)
            for i in range(len(selected_symbols)):
                ax.add_patch(plt.Rectangle((i, tdom_idx), 1, 1, fill=False, edgecolor='black', lw=2))

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TDOMAnalyzerApp()
    ex.show()
    sys.exit(app.exec_())


