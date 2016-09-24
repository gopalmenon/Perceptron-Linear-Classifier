import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Perceptron Linear Classifier 
 */
public class PerceptronLinearClassifier {
	
	public static final String FEATURE_VALUE_SEPARATOR = ":";	
	public static final int MINIMUM_SHUFFLES = 10;
	
	private Random randomNumberGenerator;
	private boolean useZeroWeights;
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
	public PerceptronLinearClassifier(boolean useZeroWeights, int numberOfEpochs, double learningRate, int numberOfFeatures) {
				
		this.randomNumberGenerator = new Random(System.currentTimeMillis());
		
		this.useZeroWeights = useZeroWeights;
		
		//Create a weights vector that includes a bias value
		this.numberOfFeatures = numberOfFeatures;
		this.weights = new ArrayList<Double>(numberOfFeatures + 1);
		for (int weightCounter = 0; weightCounter < numberOfFeatures + 1; ++weightCounter) {
			if (this.useZeroWeights) {
				//Bias will also be zero along with zero weights
				this.weights.add(Double.valueOf(0.0));
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
	 * Online training by considering training examples one at a time and updating weights and bias 
	 * @param labels
	 * @param featureVectors
	 */
	public void train(List<Integer> labels, List<String> featureVectors) {
		
		try {
			
			double dotProductOfWeightsAndFeatures = 0.0;
			for (int epochCounter = 0; epochCounter < this.numberOfEpochs; ++epochCounter) {
				
				//Loop through the training data and fine tune the weights and bias
				for (int trainingDataRowCounter = 0; trainingDataRowCounter < featureVectors.size(); ++trainingDataRowCounter) {
					
					List<Integer> featureValues = getFeatureValues(featureVectors.get(trainingDataRowCounter));
					
					//Get the dot product of weights and features
					dotProductOfWeightsAndFeatures = getDotProductOfWeightsWithInput(featureValues);
					
					//Update weights and bias if needed
					if (weightUpdateRequired(dotProductOfWeightsAndFeatures, labels.get(trainingDataRowCounter))) {
						updateWeightsAndBias(labels.get(trainingDataRowCounter), featureValues);
					}
				}
				
				if (this.numberOfEpochs > 1) {
					shuffleData(labels, featureVectors);
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
	public int predict(String featureVector) {
		
		List<Integer> featureValues = getFeatureValues(featureVector);
		if (getDotProductOfWeightsWithInput(featureValues) >= 0.0) {
			return 1;
		} else {
			return -1;
		}
		
	}
	
	
	/**
	 * @param featureVector
	 * @return the feature vector as a list of integers
	 */
	private List<Integer> getFeatureValues(String featureVector) {
		
		List<Integer> featureValues = new ArrayList<Integer>();
		int currentFeatureNumber = 0, currentFeatureValue = 0, lastFeatureNumberAdded = 0;
		String[] features = null, featureComponents = null;
		
		try {
				
			//The first element will be 1 so that it gets multiplied by the bias in the weight vector
			featureValues.add(Integer.valueOf(1));
			
			//Split the feature vector into groups of feature numbers and feature values
			features = featureVector.split(DataFileReader.WHITESPACE_REGEX);
			
			//Extract the numeric feature values
			currentFeatureNumber = 0;
			currentFeatureValue = 0;
			lastFeatureNumberAdded = 0;
			
			for (String featureGroup : features) {
				
				featureComponents = featureGroup.split(FEATURE_VALUE_SEPARATOR);
				currentFeatureNumber = Integer.parseInt(featureComponents[0]);
				currentFeatureValue = Integer.parseInt(featureComponents[1]);
				
				//Fill in the features values with zeros till the next value in the data record
				for (int featureCounter = lastFeatureNumberAdded + 1; featureCounter < currentFeatureNumber; ++featureCounter) {
					featureValues.add(Integer.valueOf(0));
				}
				
				lastFeatureNumberAdded = currentFeatureNumber;
				
				//Fill in the next value from the data file
				featureValues.add(Integer.valueOf(currentFeatureValue));
				
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
		
		int featureCounter = 0;
		double dotProduct = 0.0;
		
		for (Double weight : this.weights) {
			dotProduct += weight.doubleValue() * featureValues.get(featureCounter++).intValue();
		}
		
		return dotProduct;
		
	}
	
	/**
	 * @param dotProductOfWeightsAndFeatures
	 * @return true if weights and bias need to be updated
	 */
	protected boolean weightUpdateRequired(double dotProductOfWeightsAndFeatures, int labelValue) {
		
		if (dotProductOfWeightsAndFeatures * (double) labelValue <= 0.0) {
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
				this.weights.set(weightsCounter, weight.doubleValue() +  (this.learningRate * (double) label));
			} else {
				//Update the weight
				this.weights.set(weightsCounter, Double.valueOf(weight.doubleValue() +  (this.learningRate * (double) label * featureValues.get(weightsCounter).doubleValue())));
			}
			++weightsCounter;
		}
		
	}
	
	private void shuffleData(List<Integer> labels, List<String> featureVectors) {
		
		//Generate a random number for the number of times to shuffle the data  
		int numberOfTimesToSuffle = this.randomNumberGenerator.nextInt(MINIMUM_SHUFFLES + labels.size() / 2), swapContentsWith1 = 0, swapContentsWith2 = 0, tempLabel = 0;
		String tempFeatureVector = null;
		
		//Shuffle the data
		for (int shuffleCounter = 0; shuffleCounter < numberOfTimesToSuffle; ++shuffleCounter) {
			
			//Randomly generate the row numbers to shuffle
			swapContentsWith1 = this.randomNumberGenerator.nextInt(labels.size());
			swapContentsWith2 = this.randomNumberGenerator.nextInt(labels.size());
			
			//Swap the contents
			if (swapContentsWith1 != swapContentsWith2) {
				
				tempLabel = labels.get(swapContentsWith1);
				tempFeatureVector = featureVectors.get(swapContentsWith1);
				
				labels.set(swapContentsWith1, labels.get(swapContentsWith2));
				featureVectors.set(swapContentsWith1, featureVectors.get(swapContentsWith2));
				
				labels.set(swapContentsWith2, tempLabel);
				featureVectors.set(swapContentsWith2, tempFeatureVector);
				
			}
			
		}
		
	}
}
