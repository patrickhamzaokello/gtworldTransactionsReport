import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import schedule
import time
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Tuple, Dict


class TransactionAnalyzer:
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize the analyzer with configuration"""
        self.load_config(config_path)
        self.setup_logging()
        
    def load_config(self, config_path: str):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
    def setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            filename=self.config['logging']['file_path'],
            level=getattr(logging, self.config['logging']['level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self, year: int) -> pd.DataFrame:
        """Load transaction data for specified year"""
        try:
            # Adjust this based on your data source (SQL, CSV, etc.)
            if self.config['data_source']['type'] == 'csv':
                file_pattern = self.config['data_source']['file_pattern'].format(year=year)
                df = pd.read_csv(file_pattern)
            elif self.config['data_source']['type'] == 'sql':
                # Add SQL connection logic here
                pass
            
            # Validate data
            required_columns = [
                'TransID', 'CustomerName', 'CustomerAccount', 'Amount', 
                'Currency', 'TransType', 'PaymentDate', 'Status', 
                'ForexType', 'ForexRate', 'AccountType'
            ]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert PaymentDate to datetime
            df['PaymentDate'] = pd.to_datetime(df['PaymentDate'])
            
            # Add derived columns
            df['u_month_year'] = df['PaymentDate'].dt.strftime('%Y-%m')
            df['u_quarter_year'] = df['PaymentDate'].dt.to_period('Q').astype(str)
            df['hour_of_day'] = df['PaymentDate'].dt.hour
            df['day_of_week'] = df['PaymentDate'].dt.dayofweek
            df['is_weekend'] = df['PaymentDate'].dt.dayofweek.isin([5, 6])
            df['is_month_end'] = df['PaymentDate'].dt.is_month_end
            df['day_of_week_name'] = df['PaymentDate'].dt.day_name()
            df['month'] = df['PaymentDate'].dt.month
            df['u_month'] = df['PaymentDate'].dt.strftime('%B')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data for year {year}: {str(e)}")
            raise

    def currency_forex_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Analyze currency distributions and forex impact
        """
        # Currency distribution analysis
        currency_stats = {
            'volume_by_currency': df.groupby('Currency').size().to_dict(),
            'value_by_currency': df.groupby('Currency')['Amount'].sum().to_dict(),
            'avg_transaction_by_currency': df.groupby('Currency')['Amount'].mean().to_dict()
        }
        
        # Forex analysis
        forex_stats = {
            'avg_forex_rate': df.groupby('Currency')['ForexRate'].mean().to_dict(),
            'forex_volatility': df.groupby('Currency')['ForexRate'].std().to_dict(),
        }
        
        # Calculate forex impact
        df['ValueInBaseCurrency'] = df['Amount'] * df['ForexRate']
        forex_impact = {
            'total_value_original': df['Amount'].sum(),
            'total_value_converted': df['ValueInBaseCurrency'].sum(),
            'forex_impact_percentage': ((df['ValueInBaseCurrency'].sum() - df['Amount'].sum()) 
                                    / df['Amount'].sum() * 100)
        }
        
        return {
            'currency_stats': currency_stats,
            'forex_stats': forex_stats,
            'forex_impact': forex_impact
        }

    def account_type_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Analyze performance by account type
        """
        account_metrics = {}
        
        # Basic metrics by account type
        account_metrics['transaction_counts'] = df.groupby('AccountType').size().to_dict()
        account_metrics['total_value'] = df.groupby('AccountType')['Amount'].sum().to_dict()
        account_metrics['avg_transaction_value'] = df.groupby('AccountType')['Amount'].mean().to_dict()
        
        # Success rate by account type
        success_rate = (df[df['Status'] == 'SUCCESS'].groupby('AccountType').size() / 
                    df.groupby('AccountType').size() * 100)
        account_metrics['success_rate'] = success_rate.to_dict()
        
        # Monthly growth rate by account type
        monthly_values = df.pivot_table(
            index='u_month_year',
            columns='AccountType',
            values='Amount',
            aggfunc='sum'
        ).fillna(0)
        
        growth_rates = monthly_values.pct_change() * 100
        account_metrics['avg_monthly_growth'] = growth_rates.mean().to_dict()
        
        return account_metrics

    def risk_pattern_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Identify unusual patterns and potential risks
        """
        risk_patterns = {}
        
        # Amount outlier analysis
        Q1 = df['Amount'].quantile(0.25)
        Q3 = df['Amount'].quantile(0.75)
        IQR = Q3 - Q1
        amount_outliers = df[(df['Amount'] < (Q1 - 1.5 * IQR)) | 
                            (df['Amount'] > (Q3 + 1.5 * IQR))]
        
        risk_patterns['amount_outliers'] = {
            'count': len(amount_outliers),
            'percentage': len(amount_outliers) / len(df) * 100,
            'total_value': amount_outliers['Amount'].sum()
        }
        
        # Frequency analysis
        daily_customer_transactions = df.groupby(['PaymentDate', 'CustomerAccount']).size()
        avg_daily_transactions = daily_customer_transactions.mean()
        std_daily_transactions = daily_customer_transactions.std()
        
        frequent_transactions = daily_customer_transactions[
            daily_customer_transactions > (avg_daily_transactions + 2 * std_daily_transactions)
        ]
        
        risk_patterns['unusual_frequency'] = {
            'threshold': avg_daily_transactions + 2 * std_daily_transactions,
            'instances': len(frequent_transactions),
            'customers_involved': len(frequent_transactions.index.get_level_values('CustomerAccount').unique())
        }
        
        return risk_patterns

    def operational_efficiency_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Analyze operational efficiency by transaction type
        """
        efficiency_metrics = {}
        
        # Success rate by transaction type
        success_rate = (df[df['Status'] == 'SUCCESS'].groupby('TransType').size() / 
                    df.groupby('TransType').size() * 100)
        efficiency_metrics['success_rate_by_type'] = success_rate.to_dict()
        
        # Peak hours by transaction type
        peak_hours = df.groupby(['TransType', 'hour_of_day']).size()
        efficiency_metrics['peak_hours'] = {
            trans_type: peak_hours.loc[trans_type].idxmax()
            for trans_type in df['TransType'].unique()
        }
        
        # Failure analysis
        failed_transactions = df[df['Status'] != 'SUCCESS']
        # Convert tuple keys to strings
        failure_patterns = failed_transactions.groupby(['TransType', 'Status']).size()
        efficiency_metrics['failure_patterns'] = {
            f"{key[0]}_{key[1]}": value 
            for key, value in failure_patterns.to_dict().items()
        }
        
        return efficiency_metrics

    def weekend_weekday_comparison(self, df: pd.DataFrame) -> Dict:
        """
        Compare weekend vs weekday transaction patterns
        """
        comparison = {}
        
        # Volume comparison
        volume_comp = df.groupby('is_weekend').size()
        comparison['transaction_volume'] = {
            'weekend_avg': volume_comp[True] / (len(df['PaymentDate'].unique()) / 7 * 2),
            'weekday_avg': volume_comp[False] / (len(df['PaymentDate'].unique()) / 7 * 5)
        }
        
        # Value comparison
        value_comp = df.groupby('is_weekend')['Amount'].agg(['sum', 'mean'])
        comparison['transaction_value'] = {
            'weekend': value_comp.loc[True].to_dict(),
            'weekday': value_comp.loc[False].to_dict()
        }
        
        # Success rate comparison
        success_rate = df.groupby('is_weekend')['Status'].apply(
            lambda x: (x == 'SUCCESS').mean() * 100
        )
        comparison['success_rate'] = {
            'weekend': success_rate[True],
            'weekday': success_rate[False]
        }
        
        return comparison

    def month_end_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Analyze month-end transaction patterns
        """
        month_end_patterns = {}
        
        # Volume comparison
        volume_comp = df.groupby('is_month_end').size()
        month_end_patterns['volume_comparison'] = {
            'month_end': volume_comp.get(True, 0),
            'regular_days': volume_comp.get(False, 0),
            'volume_ratio': volume_comp.get(True, 0) / volume_comp.get(False, 0)
        }
        
        # Value comparison
        value_comp = df.groupby('is_month_end')['Amount'].agg(['sum', 'mean', 'std'])
        month_end_patterns['value_comparison'] = {
            'month_end': value_comp.loc[True].to_dict() if True in value_comp.index else None,
            'regular_days': value_comp.loc[False].to_dict() if False in value_comp.index else None
        }
        
        # Success rate comparison
        success_rate = df.groupby('is_month_end')['Status'].apply(
            lambda x: (x == 'SUCCESS').mean() * 100
        )
        month_end_patterns['success_rate'] = {
            'month_end': success_rate.get(True),
            'regular_days': success_rate.get(False)
        }
        
        return month_end_patterns

    def generate_full_analysis_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate a comprehensive analysis report
        """
        report = {
            'currency_forex_analysis': self.currency_forex_analysis(df),
            'account_type_analysis': self.account_type_analysis(df),
            'risk_patterns': self.risk_pattern_analysis(df),
            'operational_efficiency': self.operational_efficiency_analysis(df),
            'weekend_comparison': self.weekend_weekday_comparison(df),
            'month_end_patterns': self.month_end_analysis(df)
        }
        
        return report
    def generate_visualizations(self, df: pd.DataFrame, year: int, output_dir: str):
        """Generate and save visualizations"""
        try:
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Transaction Volume by Month
            plt.figure(figsize=(12, 6))
            monthly_volume = df.groupby('u_month')['TransID'].count()
            monthly_volume.plot(kind='bar')
            plt.title(f'Transaction Volume by Month - {year}')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/monthly_volume_{year}.png')
            plt.close()
            
            # Currency Distribution
            plt.figure(figsize=(10, 6))
            df['Currency'].value_counts().plot(kind='pie', autopct='%1.1f%%')
            plt.title(f'Currency Distribution - {year}')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/currency_distribution_{year}.png')
            plt.close()
            
            # More visualizations can be added here
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations for year {year}: {str(e)}")
            raise

    def save_results(self, results: dict, year: int):
        """Save analysis results to file"""
        output_dir = self.config['output']['directory']
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        with open(f'{output_dir}/analysis_results_{year}.json', 'w') as f:
            json.dump(results, f, indent=4, default=str)
            
        self.logger.info(f"Analysis results for {year} saved successfully")

    def send_email_report(self, year: int, results: dict):
        """Send email with analysis results"""
        try:
            smtp_config = self.config['email']
            
            msg = MIMEMultipart()
            msg['Subject'] = f'Transaction Analysis Report - {year}'
            msg['From'] = smtp_config['from']
            msg['To'] = smtp_config['to']
            
            # Create HTML body with key findings
            html_body = f"""
            <h2>Transaction Analysis Report - {year}</h2>
            <h3>Key Findings:</h3>
            <ul>
                <li>Total Transactions: {results['total_transactions']:,}</li>
                <li>Total Transaction Value: {results['total_value']:,.2f}</li>
                <li>Success Rate: {results['success_rate']:.2%}</li>
            </ul>
            """
            
            msg.attach(MIMEText(html_body, 'html'))
            
            # Attach JSON results
            with open(f"{self.config['output']['directory']}/analysis_results_{year}.json", 'rb') as f:
                attachment = MIMEApplication(f.read(), _subtype='json')
                attachment.add_header('Content-Disposition', 'attachment', 
                                   filename=f'analysis_results_{year}.json')
                msg.attach(attachment)
            
            # Send email
            with smtplib.SMTP(smtp_config['server'], smtp_config['port']) as server:
                server.starttls()
                server.login(smtp_config['username'], smtp_config['password'])
                server.send_message(msg)
                
            self.logger.info(f"Email report for {year} sent successfully")
            
        except Exception as e:
            self.logger.error(f"Error sending email report for {year}: {str(e)}")
            raise

    def run_analysis(self, year: int = None):
        """Run the complete analysis for a specific year"""
        try:
            # If year is not specified, use previous year
            if year is None:
                year = datetime.now().year - 1
                
            self.logger.info(f"Starting analysis for year {year}")
            
            # Load data
            df = self.load_data(year)
            
            # Generate analysis using previous functions
            results = self.generate_full_analysis_report(df)  # From previous code
            
            # Add high-level metrics
            results.update({
                'total_transactions': len(df),
                'total_value': df['Amount'].sum(),
                'success_rate': (df['Status'] == 'SUCCESS').mean(),
                'analysis_timestamp': datetime.now().isoformat()
            })
            
            # Generate visualizations
            self.generate_visualizations(
                df, 
                year, 
                f"{self.config['output']['directory']}/visualizations"
            )
            
            # Save results
            self.save_results(results, year)
            
            # Send email report
            # if self.config['email']['enabled']:
            #     self.send_email_report(year, results)
            
            self.logger.info(f"Analysis for year {year} completed successfully")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in analysis for year {year}: {str(e)}")
            raise

# def setup_scheduler(analyzer: TransactionAnalyzer):
#     """Set up scheduled runs"""
#     schedule.every().day.at("01:00").do(analyzer.run_analysis)  # Daily run at 1 AM
    
#     # Also add monthly and yearly schedules if needed
#     schedule.every().month.at("1-00:00").do(analyzer.run_analysis)  # Monthly run
    
#     while True:
#         schedule.run_pending()
#         time.sleep(60)



def main():
    """Main function to start the automated analysis"""    
    # Initialize analyzer
    analyzer = TransactionAnalyzer()
    
    # Run analysis for previous year if needed
    previous_year = datetime.now().year - 1
    if not os.path.exists(f"analysis_output/analysis_results_{previous_year}.json"):
        analyzer.run_analysis(previous_year)

    
    # Start scheduler
    # setup_scheduler(analyzer)

if __name__ == "__main__":
    main()