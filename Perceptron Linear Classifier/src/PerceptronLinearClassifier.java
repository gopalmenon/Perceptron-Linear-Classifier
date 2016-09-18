import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;


/**
 * Perceptron Linear Classifier 
 */
public class PerceptronLinearClassifier {
	
	public static String WHITESPACE_REGEX = "\\s";
	public static final String LABEL_DATA_SEPARATOR = ":";	
	
	private Random randomNumberGenerator;
	private boolean useZeroWeights;
	private List<String> rawTrainingData;
	private List<String> rawTestingData;
	private List<Double> weights;
	private int numberOfEpochs;
	private int numberOfFeatures;
	private double learningRate;
	
	/**
	 * Constructor
	 * @param useZeroWeights
	 * @param numberOfEpochs
	 * @param trainingDataFileName
	 * @param testingDataFileName
	 */
	public PerceptronLinearClassifier(boolean useZeroWeights, int numberOfEpochs, double learningRate, String trainingDataFileName, String testingDataFileName) {
				
		this.randomNumberGenerator = new Random(System.currentTimeMillis());
		
		this.useZeroWeights = useZeroWeights;
		
		//Create a weights vector that includes a bias value
		this.numberOfFeatures = loadRawData(trainingDataFileName, testingDataFileName);
		this.weights = new ArrayList<Double>(numberOfFeatures + 1);
		for (int weightCounter = 0; weightCounter < numberOfFeatures + 1; ++weightCounter) {
			if (this.useZeroWeights) {
				//First term is the bias
				if (weightCounter == 0) {
					this.weights.add(Double.valueOf(this.randomNumberGenerator.nextDouble()));
				} else {
					this.weights.add(Double.valueOf(0.0));
				}
			} else {
				this.weights.add(Double.valueOf(this.randomNumberGenerator.nextDouble()));
			}
		}
		
		if (numberOfEpochs <= 0) {
			throw new IllegalArgumentException("Number of epochs parameter must be greater than zero. Parameter value " + Integer.valueOf(numberOfEpochs).toString() + " is not valid.");
		}
		
		this.numberOfEpochs = numberOfEpochs;
		this.learningRate = learningRate;
		
	}
	
	/**
	 * Load raw data and return the number of features
	 * @param trainingDataFileName
	 * @param testingDataFileName
	 * @return
	 */
	private int loadRawData(String trainingDataFileName, String testingDataFileName) {
		
		int numberOfTrainingDataFeatures = Integer.MIN_VALUE, numberOfTestingDataFeatures = Integer.MIN_VALUE, currentLastLabelColumnNumber = 0;
		try {

			//Find number of features in training data
			this.rawTrainingData = DataFileReader.getDataFileContents(trainingDataFileName);
			for(String fileLine : this.rawTrainingData) {
				String[] rawDataColumns = fileLine.split(WHITESPACE_REGEX);
				String[] lastLabelAndFeature = rawDataColumns[rawDataColumns.length - 1].split(LABEL_DATA_SEPARATOR);
				currentLastLabelColumnNumber = Integer.parseInt(lastLabelAndFeature[0]);
				if (currentLastLabelColumnNumber > numberOfTrainingDataFeatures) {
					numberOfTrainingDataFeatures = currentLastLabelColumnNumber;
				}
			}

			//Find number of features in testing data
			this.rawTestingData = DataFileReader.getDataFileContents(testingDataFileName);
			for(String fileLine : this.rawTestingData) {
				String[] rawDataColumns = fileLine.split(WHITESPACE_REGEX);
				String[] lastLabelAndFeature = rawDataColumns[rawDataColumns.length - 1].split(LABEL_DATA_SEPARATOR);
				currentLastLabelColumnNumber = Integer.parseInt(lastLabelAndFeature[0]);
				if (currentLastLabelColumnNumber > numberOfTestingDataFeatures) {
					numberOfTestingDataFeatures = currentLastLabelColumnNumber;
				}
			}
			
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(0);
		} catch (NumberFormatException e) {
			e.printStackTrace();
			System.exit(0);
		}
		
		return Math.max(numberOfTrainingDataFeatures, numberOfTestingDataFeatures);
	}

	
	/**
	 * Online training by considering training examples one at a time and updating weights and bias 
	 */
	public void train() {
		
		try {
			double dotProductOfWeightsAndFeatures = 0.0;
			for (int epochCounter = 0; epochCounter < this.numberOfEpochs; ++epochCounter) {
				
				//Loop through the training data and fine tune the weights and bias
				String trainingDataRow = null;
				int label = 0;
				for (int trainingDataRowCounter = 0; trainingDataRowCounter < this.rawTrainingData.size(); ++trainingDataRowCounter) {
					
					//Get label and feature values for training data row
					trainingDataRow = this.rawTrainingData.get(trainingDataRowCounter);
					String[] labelAndFeatures = trainingDataRow.split(WHITESPACE_REGEX);
					label = Integer.parseInt(labelAndFeatures[0]);
					List<Integer> featureValues = getFeatureValues(labelAndFeatures);
					
					//Get the dot product of weights and features
					dotProductOfWeightsAndFeatures = getDotProductOfWeightsWithInput(featureValues);
					
					//Update weights and bias if needed
					if (weightUpdateRequired(dotProductOfWeightsAndFeatures, label)) {
						updateWeightsAndBias(label, featureValues);
					}
				}
				
			}
		} catch (NumberFormatException e) {
			System.err.println("Invalid label value in training data.");
			System.exit(0);
		}
		
	}
	
	/**
	 * @param testingDataRow
	 * @return the label that will be the sign of the dot product between the testing data and the input features 
	 */
	public int predict(String testingDataRow) {
		
		String[] labelAndFeatures = testingDataRow.split(WHITESPACE_REGEX);
		List<Integer> featureValues = getFeatureValues(labelAndFeatures);
		if (getDotProductOfWeightsWithInput(featureValues) >= 0.0) {
			return 1;
		} else {
			return -1;
		}
		
	}
	
	
	private List<Integer> getFeatureValues(String[] labelAndFeatures) {
		
		List<Integer> featureValues = new ArrayList<Integer>();
		int loopIndex = 0, currentFeatureNumber = 0, currentFeatureValue = 0, lastFeatureNumberAdded = 0;
		String[] featureAndValue = null;
		
		try {
			
			//Loop through the features in a data row
			for (String rawFeatures : labelAndFeatures) {
				
				//The first element will be 1 so that it gets multiplied by the bias in the weight vector
				if (loopIndex == 0) {
					featureValues.add(Integer.valueOf(1));
				} else {
					featureAndValue = rawFeatures.split(LABEL_DATA_SEPARATOR);
					currentFeatureNumber = Integer.parseInt(featureAndValue[0]);
					currentFeatureValue = Integer.parseInt(featureAndValue[1]);
					
					//Fill in the features values with zeros till the next value in the data record
					for (int featureCounter = lastFeatureNumberAdded + 1; featureCounter < currentFeatureNumber; ++featureCounter) {
						featureValues.add(Integer.valueOf(0));
					}
					
					lastFeatureNumberAdded = currentFeatureNumber;
					
					//Fill in the next value from the data file
					featureValues.add(Integer.valueOf(currentFeatureValue));
					
				}
				++loopIndex;
			}
			
			//Add the rest of the features with zero values
			for (int featureCounter = lastFeatureNumberAdded + 1; featureCounter <= this.numberOfFeatures; ++featureCounter) {
				featureValues.add(Integer.valueOf(0));
			}

		} catch (NumberFormatException e) {
				System.err.println("Invalid feature value found.");
				System.exit(0);
		}
			
		
		return featureValues;
		
	}
	
	/**
	 * @param featureValues
	 * @return dot product of the weights with the feature values
	 */
	private double getDotProductOfWeightsWithInput(List<Integer> featureValues) {
		
		int weightCounter = 0;
		double dotProduct = 0.0;
		
		for (Double weight : this.weights) {
			dotProduct += weight.doubleValue() * featureValues.get(weightCounter++).intValue();
		}
		
		return dotProduct;
		
	}
	
	/**
	 * @param dotProductOfWeightsAndFeatures
	 * @return true if weights and bias need to be updated
	 */
	protected boolean weightUpdateRequired(double dotProductOfWeightsAndFeatures, int labelValue) {
		
		if (dotProductOfWeightsAndFeatures * labelValue <= 0) {
			return true;
		} else {
			return false;
		}
		
	}
	
	/**
	 * Update the weights and bias 
	 * @param label
	 * @param featureValues
	 */
	private void updateWeightsAndBias(int label, List<Integer> featureValues) {
		
		int weightsCounter = 0;
		for (Double weight : this.weights) {
			
			if (weightsCounter == 0) {
				//Update the bias
				this.weights.set(weightsCounter, weight.doubleValue() + (double) (this.learningRate * label));
			} else {
				//Update the weight
				this.weights.set(weightsCounter, Double.valueOf(weight.doubleValue() + (double) (this.learningRate * label * featureValues.get(weightsCounter).intValue())));
			}
			++weightsCounter;
		}
		
	}
}
