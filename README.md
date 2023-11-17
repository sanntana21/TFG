# FORECASTING SERIES TEMPORALES

Forecasting de series temporales utilizando distintos metodos de preprocesado y una implementacion de LSTM como modelo de predicción. 

## Install required Python packages:
Download and the unzip the proyect. Then run:
```bash 
pip install -r requirements.txt
pip install tensorflow
pip install plotly_express
```
Un zip the Resources dir in the same dir where you unziped the proyect.
Your proyect main dir should look like this:

- METS_predictions
- results_analysis
- utils.py
- requirements.txt
- Resources
   - Agregado
   - Individual
   - Resultados


## Usage

Run the METs predictions script:
```bash 
python3 ./METS_predictions
```
In order of getting CSVs with results you can run:
```bash 
python3 ./results_analysis
```
Make sure you have satisfied all the dependencies mentioned in the `requirements.txt` file before running the script.

## Contributing

If you'd like to contribute to this project, please follow these steps:

1. Fork the repository
2. Create a new branch: `git checkout -b feature-name`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Create a pull request

## License

This project is licensed  - see the [LICENSE.md](LICENSE.md) file for details.
