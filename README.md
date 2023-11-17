# FORECASTING SERIES TEMPORALES

Forecasting de series temporales utilizando distintos metodos de preprocesado y una implementacion de LSTM como modelo de predicción. 

## Install required Python packages:
Download and unzip the project. Then run the following commands:

```bash 
pip install -r requirements.txt
pip install tensorflow
pip install plotly_express
```
To obtain the data resources, you will need to request access, as the original data is part of the [POSTCOVID-ai poryect](https://projects.ugr.es/postcovid-ai).
Unzip the "Resources" directory in the same location where you unzipped the project. Your project's main directory should resemble the following:

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
To obtain CSVs with results, run:
```bash 
python3 ./results_analysis
```
Ensure that you have satisfied all the dependencies mentioned in the requirements.txt file before running the script.

## Contributing

If you'd like to contribute to this project, please follow these steps:

1. Fork the repository
2. Create a new branch: `git checkout -b feature-name`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Create a pull request

## License

This project is licensed  - see the [LICENSE.md](LICENSE.md) file for details.
