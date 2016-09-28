import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;


public class RunClassfier {

	public static final String TABLE_2_DATA_FILE = "data/table2";
	public static final String TRAINING_DATA_FILE = "data/a5a.train";
	public static final String TESTING_DATA_FILE = "data/a5a.test";
	public static final String TRAINING_DATA_FILE_2D = "data/2dTrain.txt";
	public static final String TESTING_DATA_FILE_2D = "data/2dTest.txt";
	
	public static final int CROSS_VALIDATION_FOLDS = 6;

	public static final double[] TEST_LEARNING_RATES = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
	public static final double[] TEST_MYU_VALUES = {1.0, 2.0, 3.0, 4.0, 5.0};
	
	public static final int[] SINGLE_EPOCH = {1};
	public static final int[] MULTIPLE_EPOCHS = {3, 5};
	
	public static void main(String[] args) {
		
		RunClassfier runClassifier = new RunClassfier();
		runClassifier.runTests();

	}
	
	private void runTests() {
		
		try {
			
			LabelsAndFeatures table2LabelsAndFeatures = DataFileReader.getDataFileContents(TABLE_2_DATA_FILE);
			LabelsAndFeatures trainingLabelsAndFeatures = DataFileReader.getDataFileContents(TRAINING_DATA_FILE);
			LabelsAndFeatures testingLabelsAndFeatures = DataFileReader.getDataFileContents(TESTING_DATA_FILE);	
			int numberOfTrainingDataFeatures = getNumberOfFeatures(trainingLabelsAndFeatures.getFeatureVectors()), numberOfTestingDataFeatures = getNumberOfFeatures(testingLabelsAndFeatures.getFeatureVectors());
			int numberOfFeatures = Math.max(numberOfTrainingDataFeatures, numberOfTestingDataFeatures);
			
			runExperiment1(table2LabelsAndFeatures, getNumberOfFeatures(getCleanedTable2LabelsAndFeatures(table2LabelsAndFeatures.getFeatureVectors())));
			
			runExperiment2(trainingLabelsAndFeatures, testingLabelsAndFeatures, numberOfFeatures);
			
			runExperiment3(trainingLabelsAndFeatures, testingLabelsAndFeatures, numberOfFeatures);
			
			runExperiment4(trainingLabelsAndFeatures, testingLabelsAndFeatures, numberOfFeatures);
			
		} catch (IOException e) {
			System.err.println("Error while trying to extract labels from test data.");
			e.printStackTrace();
		}
		
	}
	
	/**
	 * Run Experiment 1
	 * @param trainingLabelsAndFeatures
	 * @param testingLabelsAndFeatures
	 * @param numberOfFeatures
	 */
	private void runExperiment1(LabelsAndFeatures trainingLabelsAndFeatures, int numberOfFeatures) {
		
		//Experiment 1
		PerceptronLinearClassifier classifier = new PerceptronLinearClassifier(true, 1, 1.0, numberOfFeatures);
		System.out.println("3.3.1 Number of mistakes made is " + classifier.train(trainingLabelsAndFeatures.getLabels(), getCleanedTable2LabelsAndFeatures(trainingLabelsAndFeatures.getFeatureVectors())));
		System.out.println("3.3.1 Weight vector is " + classifier.getWeightVector().toString());

	}
	
	/**
	 * Run Experiment 2
	 * @param trainingLabelsAndFeatures
	 * @param testingLabelsAndFeatures
	 * @param numberOfFeatures
	 */
	private void runExperiment2(LabelsAndFeatures trainingLabelsAndFeatures, LabelsAndFeatures testingLabelsAndFeatures, int numberOfFeatures) {
		
		//Experiment 2
		double bestPerceptronLearningRate = getPerceptronLearningRateByCrossValidation(CROSS_VALIDATION_FOLDS, trainingLabelsAndFeatures.getLabels(), trainingLabelsAndFeatures.getFeatureVectors(), numberOfFeatures, SINGLE_EPOCH).getLearningRate();
		System.out.println("\n3.3.2 Perceptron Learning Rate used : " + bestPerceptronLearningRate);

		PerceptronLinearClassifier classifier = new PerceptronLinearClassifier(false, 1, bestPerceptronLearningRate, numberOfFeatures);
		System.out.println("3.3.2 Number of Perceptron mistakes made is " + classifier.train(trainingLabelsAndFeatures.getLabels(), trainingLabelsAndFeatures.getFeatureVectors()));
		System.out.println("3.3.2 Perceptron Weight vector is " + classifier.getWeightVector().toString());
		
		List<Integer> predictions = new ArrayList<Integer>();
		for (String trainingDataVector : trainingLabelsAndFeatures.getFeatureVectors()) {
			predictions.add(Integer.valueOf(classifier.predict(trainingDataVector)));
		}
		
		ClassifierMetrics classifierMetrics = new ClassifierMetrics(trainingLabelsAndFeatures.getLabels(), predictions);
		
		System.out.println("3.3.2 Perceptron Accuracy with training data: " + classifierMetrics.getAccuracy());
		
		predictions = new ArrayList<Integer>();
		for (String testingDataVector : testingLabelsAndFeatures.getFeatureVectors()) {
			predictions.add(Integer.valueOf(classifier.predict(testingDataVector)));
		}
		
		classifierMetrics = new ClassifierMetrics(testingLabelsAndFeatures.getLabels(), predictions);
		
		System.out.println("3.3.2 Perceptron Accuracy with testing data: " + classifierMetrics.getAccuracy());

		LearningRateAndMyu learningRateAndMyu = getMarginPerceptronLearningRateAndMyuByCrossValidation(CROSS_VALIDATION_FOLDS, trainingLabelsAndFeatures.getLabels(), trainingLabelsAndFeatures.getFeatureVectors(), numberOfFeatures, SINGLE_EPOCH);
		System.out.println("\n3.3.2 Margin Perceptron Learning Rate used : " + learningRateAndMyu.getLearningRate() + ", and Myu used is " + learningRateAndMyu.getMyu());
		MarginPerceptron marginClassifier = new MarginPerceptron(false, 1, learningRateAndMyu.getLearningRate(), numberOfFeatures, learningRateAndMyu.getMyu());
		System.out.println("3.3.2 Number of Margin Perceptron mistakes made is " + marginClassifier.train(trainingLabelsAndFeatures.getLabels(), trainingLabelsAndFeatures.getFeatureVectors()));
		System.out.println("3.3.2 Margin Perceptron Weight vector is " + marginClassifier.getWeightVector().toString());
		
		predictions = new ArrayList<Integer>();
		for (String trainingDataVector : trainingLabelsAndFeatures.getFeatureVectors()) {
			predictions.add(Integer.valueOf(marginClassifier.predict(trainingDataVector)));
		}
		
		classifierMetrics = new ClassifierMetrics(trainingLabelsAndFeatures.getLabels(), predictions);
		
		System.out.println("3.3.2 Margin Perceptron Accuracy with training data: " + classifierMetrics.getAccuracy());
		
		predictions = new ArrayList<Integer>();
		for (String testingDataVector : testingLabelsAndFeatures.getFeatureVectors()) {
			predictions.add(Integer.valueOf(marginClassifier.predict(testingDataVector)));
		}
		
		classifierMetrics = new ClassifierMetrics(testingLabelsAndFeatures.getLabels(), predictions);
		
		System.out.println("3.3.2 Margin Perceptron Accuracy with testing data: " + classifierMetrics.getAccuracy());
	}
	
	/**
	 * Run Experiment 3
	 * @param trainingLabelsAndFeatures
	 * @param testingLabelsAndFeatures
	 * @param numberOfFeatures
	 */
	private void runExperiment3(LabelsAndFeatures trainingLabelsAndFeatures, LabelsAndFeatures testingLabelsAndFeatures, int numberOfFeatures) {
		
		//Experiment 3
		LearningRateAndMyu learningRateAndMyu = getPerceptronLearningRateByCrossValidation(CROSS_VALIDATION_FOLDS, trainingLabelsAndFeatures.getLabels(), trainingLabelsAndFeatures.getFeatureVectors(), numberOfFeatures, MULTIPLE_EPOCHS);
		System.out.println("\n3.3.3 Perceptron Learning Rate used : " + learningRateAndMyu.getLearningRate() + " for " + learningRateAndMyu.getEpochs() + " epochs.");

		PerceptronLinearClassifier classifier = new PerceptronLinearClassifier(false, learningRateAndMyu.getEpochs(), learningRateAndMyu.getLearningRate(), numberOfFeatures);
		System.out.println("3.3.3 Number of Perceptron mistakes made is " + classifier.train(trainingLabelsAndFeatures.getLabels(), trainingLabelsAndFeatures.getFeatureVectors()));
		System.out.println("3.3.3 Perceptron Weight vector is " + classifier.getWeightVector().toString());
		
		List<Integer> predictions = new ArrayList<Integer>();
		for (String trainingDataVector : trainingLabelsAndFeatures.getFeatureVectors()) {
			predictions.add(Integer.valueOf(classifier.predict(trainingDataVector)));
		}
		
		ClassifierMetrics classifierMetrics = new ClassifierMetrics(trainingLabelsAndFeatures.getLabels(), predictions);
		
		System.out.println("3.3.3 Perceptron Accuracy on training data: " + classifierMetrics.getAccuracy());
		
		predictions = new ArrayList<Integer>();
		for (String testingDataVector : testingLabelsAndFeatures.getFeatureVectors()) {
			predictions.add(Integer.valueOf(classifier.predict(testingDataVector)));
		}
		
		classifierMetrics = new ClassifierMetrics(testingLabelsAndFeatures.getLabels(), predictions);
		
		System.out.println("3.3.3 Perceptron Accuracy on testing data: " + classifierMetrics.getAccuracy());

		learningRateAndMyu = getMarginPerceptronLearningRateAndMyuByCrossValidation(CROSS_VALIDATION_FOLDS, trainingLabelsAndFeatures.getLabels(), trainingLabelsAndFeatures.getFeatureVectors(), numberOfFeatures, MULTIPLE_EPOCHS);
		System.out.println("\n3.3.3 Margin Perceptron Learning Rate used : " + learningRateAndMyu.getLearningRate() + ", and Myu used is " + learningRateAndMyu.getMyu() + ", for " + learningRateAndMyu.getEpochs() + " epochs.");
		MarginPerceptron marginClassifier = new MarginPerceptron(false, learningRateAndMyu.getEpochs(), learningRateAndMyu.getLearningRate(), numberOfFeatures, learningRateAndMyu.getMyu());
		System.out.println("3.3.3 Number of Margin Perceptron mistakes made is " + marginClassifier.train(trainingLabelsAndFeatures.getLabels(), trainingLabelsAndFeatures.getFeatureVectors()));
		System.out.println("3.3.3 Margin Perceptron Weight vector is " + marginClassifier.getWeightVector().toString());
		
		predictions = new ArrayList<Integer>();
		for (String trainingDataVector : trainingLabelsAndFeatures.getFeatureVectors()) {
			predictions.add(Integer.valueOf(marginClassifier.predict(trainingDataVector)));
		}
		
		classifierMetrics = new ClassifierMetrics(trainingLabelsAndFeatures.getLabels(), predictions);
		
		System.out.println("3.3.3 Margin Perceptron Accuracy on training data: " + classifierMetrics.getAccuracy());
		
		predictions = new ArrayList<Integer>();
		for (String testingDataVector : testingLabelsAndFeatures.getFeatureVectors()) {
			predictions.add(Integer.valueOf(marginClassifier.predict(testingDataVector)));
		}
		
		classifierMetrics = new ClassifierMetrics(testingLabelsAndFeatures.getLabels(), predictions);
		
		System.out.println("3.3.3 Margin Perceptron Accuracy on testing data: " + classifierMetrics.getAccuracy());
	}
	
	/**
	 * Run Experiment 4
	 * @param trainingLabelsAndFeatures
	 * @param testingLabelsAndFeatures
	 * @param numberOfFeatures
	 */
	private void runExperiment4(LabelsAndFeatures trainingLabelsAndFeatures, LabelsAndFeatures testingLabelsAndFeatures, int numberOfFeatures) {
		
		//Experiment 4
		LearningRateAndMyu learningRateAndMyu = getAggressiveMarginPerceptronLearningRateAndMyuByCrossValidation(CROSS_VALIDATION_FOLDS, trainingLabelsAndFeatures.getLabels(), trainingLabelsAndFeatures.getFeatureVectors(), numberOfFeatures, SINGLE_EPOCH);
		System.out.println("\n3.3.4 Aggressive Margin Perceptron Myu used is " + learningRateAndMyu.getMyu() + ", for " + learningRateAndMyu.getEpochs() + " epochs.");
		AggressiveMarginPerceptron aggressiveMarginClassifier = new AggressiveMarginPerceptron(false, learningRateAndMyu.getEpochs(), numberOfFeatures, learningRateAndMyu.getMyu());
		System.out.println("3.3.4 Number of Aggressive Margin Perceptron mistakes made is " + aggressiveMarginClassifier.train(trainingLabelsAndFeatures.getLabels(), trainingLabelsAndFeatures.getFeatureVectors()));
		System.out.println("3.3.4 Aggressive Margin Perceptron Weight vector is " + aggressiveMarginClassifier.getWeightVector().toString());
		
		List<Integer> predictions = new ArrayList<Integer>();
		for (String trainingDataVector : trainingLabelsAndFeatures.getFeatureVectors()) {
			predictions.add(Integer.valueOf(aggressiveMarginClassifier.predict(trainingDataVector)));
		}
		
		ClassifierMetrics classifierMetrics = new ClassifierMetrics(trainingLabelsAndFeatures.getLabels(), predictions);
		
		System.out.println("3.3.4 Aggressive Margin Perceptron Accuracy with training data: " + classifierMetrics.getAccuracy());
		
		predictions = new ArrayList<Integer>();
		for (String testingDataVector : testingLabelsAndFeatures.getFeatureVectors()) {
			predictions.add(Integer.valueOf(aggressiveMarginClassifier.predict(testingDataVector)));
		}
		
		classifierMetrics = new ClassifierMetrics(testingLabelsAndFeatures.getLabels(), predictions);
		
		System.out.println("3.3.4 Aggressive Margin Perceptron Accuracy with testing data: " + classifierMetrics.getAccuracy());

		learningRateAndMyu = getAggressiveMarginPerceptronLearningRateAndMyuByCrossValidation(CROSS_VALIDATION_FOLDS, trainingLabelsAndFeatures.getLabels(), trainingLabelsAndFeatures.getFeatureVectors(), numberOfFeatures, MULTIPLE_EPOCHS);
		System.out.println("\n3.3.4 Aggressive Margin Perceptron Myu used is " + learningRateAndMyu.getMyu() + ", for " + learningRateAndMyu.getEpochs() + " epochs.");
		aggressiveMarginClassifier = new AggressiveMarginPerceptron(false, learningRateAndMyu.getEpochs(), numberOfFeatures, learningRateAndMyu.getMyu());
		System.out.println("3.3.4 Number of Aggressive Margin Perceptron mistakes made is " + aggressiveMarginClassifier.train(trainingLabelsAndFeatures.getLabels(), trainingLabelsAndFeatures.getFeatureVectors()));
		System.out.println("3.3.4 Aggressive Margin Perceptron Weight vector is " + aggressiveMarginClassifier.getWeightVector().toString());
		
		predictions = new ArrayList<Integer>();
		for (String trainingDataVector : trainingLabelsAndFeatures.getFeatureVectors()) {
			predictions.add(Integer.valueOf(aggressiveMarginClassifier.predict(trainingDataVector)));
		}
		
		classifierMetrics = new ClassifierMetrics(trainingLabelsAndFeatures.getLabels(), predictions);
		
		System.out.println("3.3.4 Aggressive Margin Perceptron Accuracy with training data: " + classifierMetrics.getAccuracy());
		
		predictions = new ArrayList<Integer>();
		for (String testingDataVector : testingLabelsAndFeatures.getFeatureVectors()) {
			predictions.add(Integer.valueOf(aggressiveMarginClassifier.predict(testingDataVector)));
		}
		
		classifierMetrics = new ClassifierMetrics(testingLabelsAndFeatures.getLabels(), predictions);
		
		System.out.println("3.3.4 Aggressive Margin Perceptron Accuracy with testing data: " + classifierMetrics.getAccuracy());
	}

	/**
	 * @param featureVectors
	 * @return the number of features in the feature vectors
	 */
	private int getNumberOfFeatures(List<String> featureVectors) {
		
		int currentLastLabelColumnNumber = 0, numberOfFeatures = 0;

		try {
			for(String featureVector : featureVectors) {
				String[] rawDataColumns = featureVector.split(DataFileReader.WHITESPACE_REGEX);
				String[] lastLabelAndFeature = rawDataColumns[rawDataColumns.length - 1].split(PerceptronLinearClassifier.FEATURE_VALUE_SEPARATOR);
				currentLastLabelColumnNumber = Integer.parseInt(lastLabelAndFeature[0]);
				if (currentLastLabelColumnNumber > numberOfFeatures) {
					numberOfFeatures = currentLastLabelColumnNumber;
				}
			}
		} catch (NumberFormatException e) {
			throw e;
		}
		
		return numberOfFeatures;
	}
	
	/**
	 * @param numberOfSplits
	 * @param labels
	 * @param featureVectors
	 * @return split cross validation labels and data
	 */
	private List<LabelsAndFeatures> getCrossValidationData(int numberOfSplits, List<Integer> labels, List<String> featureVectors) {
		
		Random randomNumberGenerator = new Random(0);
		
		List<LabelsAndFeatures> crossValidationData = new ArrayList<LabelsAndFeatures>(numberOfSplits);
		
		List<Integer> labelsCopy = new ArrayList<Integer>(labels);
		List<String> featureVectorsCopy = new ArrayList<String>(featureVectors);
		
		int numberOfCrossValidationDataRecords = labels.size() / numberOfSplits, randomRecordNumber = 0;
		
		//Create one less than the required number of splits
		for (int splitCounter = 0; splitCounter < numberOfSplits - 1; ++splitCounter) {
				
			List<Integer> labelsSubset = new ArrayList<Integer>(numberOfCrossValidationDataRecords);
			List<String> featureVectorsSubset = new ArrayList<String>(numberOfCrossValidationDataRecords);

			//Fill data required for cross validation split
			for (int recordCounter = 0; recordCounter < numberOfCrossValidationDataRecords; ++recordCounter) {
				
				randomRecordNumber = randomNumberGenerator.nextInt(labelsCopy.size());
				
				labelsSubset.add(labels.get(randomRecordNumber));
				labelsCopy.remove(randomRecordNumber);
				featureVectorsSubset.add(featureVectors.get(randomRecordNumber));
				featureVectorsCopy.remove(randomRecordNumber);
			}
			
			crossValidationData.add(new LabelsAndFeatures(labelsSubset, featureVectorsSubset));
			
		}
		
		//Add the remaining labels and features to the last split
		crossValidationData.add(new LabelsAndFeatures(labelsCopy, featureVectorsCopy));
		
		//Return the data
		return crossValidationData;
		
	}
	
	/**
	 * @param crossValidationFolds
	 * @param labels
	 * @param featureVectors
	 * @param numberOfFeatures
	 * @return learning rate from cross validation
	 */
	private LearningRateAndMyu getPerceptronLearningRateByCrossValidation(int crossValidationFolds, List<Integer> labels, List<String> featureVectors, int numberOfFeatures, int[] epochValues) {
		
		double currentAccuracy = 0.0, maximumAccuracy = Double.MIN_VALUE, bestLearningRate = 0.0;
		int bestNumberOfEpochs = 0;
		
		List<LabelsAndFeatures> crossValidationData = getCrossValidationData(crossValidationFolds, labels, featureVectors);
		
		for (int epochs : epochValues) {

			//Try out each of the test learning rates
			for (double learningRate : TEST_LEARNING_RATES) {
				
				//Run k-fold cross validation
				for (int crossValidationCounter = 0; crossValidationCounter < crossValidationFolds; ++crossValidationCounter) {
					
					PerceptronLinearClassifier classifier = new PerceptronLinearClassifier(false, epochs, learningRate, numberOfFeatures);
					
					//Load training and testing data
					List<Integer> trainingDataLabels = new ArrayList<Integer>();
					List<String> trainingDataFeatures = new ArrayList<String>();
					
					List<Integer> testingDataLabels = new ArrayList<Integer>();
					List<String> testingDataFeatures = new ArrayList<String>();
					
					int splitCounter = 0;
					for (LabelsAndFeatures labelsAndFeatures : crossValidationData) {
						
						if (splitCounter == crossValidationCounter) {
							testingDataLabels.addAll(labelsAndFeatures.getLabels());
							testingDataFeatures.addAll(labelsAndFeatures.getFeatureVectors());
						} else {
							trainingDataLabels.addAll(labelsAndFeatures.getLabels());
							trainingDataFeatures.addAll(labelsAndFeatures.getFeatureVectors());
						}
						
						++splitCounter;
						
					}
				
					//Train the classifier
					classifier.train(trainingDataLabels, trainingDataFeatures);
					
					//Run predictions
					List<Integer> predictions = new ArrayList<Integer>();
					for (String testingDataVector : testingDataFeatures) {
						predictions.add(Integer.valueOf(classifier.predict(testingDataVector)));
					}
					
					//Get accuracy for this learning rate
					currentAccuracy = new ClassifierMetrics(testingDataLabels, predictions).getAccuracy();
					if (currentAccuracy > maximumAccuracy) {
						maximumAccuracy = currentAccuracy;
						bestLearningRate = learningRate;
						bestNumberOfEpochs = epochs;
					}
				}
				
			}
			
		}
			
		return this.new LearningRateAndMyu(bestLearningRate, 0.0, bestNumberOfEpochs);

	}
	
	/**
	 * Helper class used for returning best value of learning rate and myu
	 *
	 */
	private class LearningRateAndMyu {
		private double learningRate;
		private double myu;
		private int epochs;
		public LearningRateAndMyu(double learningRate, double myu, int epochs) {
			this.learningRate = learningRate;
			this.myu = myu;
			this.epochs = epochs;
		}
		public double getLearningRate() {
			return learningRate;
		}
		public double getMyu() {
			return myu;
		}
		public int getEpochs() {
			return epochs;
		}
	}
	
	/**
	 * @param crossValidationFolds
	 * @param labels
	 * @param featureVectors
	 * @param numberOfFeatures
	 * @return learning rate from cross validation
	 */
	private LearningRateAndMyu getMarginPerceptronLearningRateAndMyuByCrossValidation(int crossValidationFolds, List<Integer> labels, List<String> featureVectors, int numberOfFeatures, int[] epochValues) {
		
		double currentAccuracy = 0.0, maximumAccuracy = Double.MIN_VALUE, bestLearningRate = 0.0, bestMyu = 0.0;
		int bestNumberOfEpochs = 0;
		
		List<LabelsAndFeatures> crossValidationData = getCrossValidationData(crossValidationFolds, labels, featureVectors);
				
		for (int epochs : epochValues) {
			
			for (double myu : TEST_MYU_VALUES) {
	
				//Try out each of the test learning rates
				for (double learningRate : TEST_LEARNING_RATES) {
					
					//Run k-fold cross validation
					for (int crossValidationCounter = 0; crossValidationCounter < crossValidationFolds; ++crossValidationCounter) {
						
						MarginPerceptron classifier = new MarginPerceptron(false, epochs, learningRate, numberOfFeatures, myu);
						
						//Load training and testing data
						List<Integer> trainingDataLabels = new ArrayList<Integer>();
						List<String> trainingDataFeatures = new ArrayList<String>();
						
						List<Integer> testingDataLabels = new ArrayList<Integer>();
						List<String> testingDataFeatures = new ArrayList<String>();
						
						int splitCounter = 0;
						for (LabelsAndFeatures labelsAndFeatures : crossValidationData) {
							
							if (splitCounter == crossValidationCounter) {
								testingDataLabels.addAll(labelsAndFeatures.getLabels());
								testingDataFeatures.addAll(labelsAndFeatures.getFeatureVectors());
							} else {
								trainingDataLabels.addAll(labelsAndFeatures.getLabels());
								trainingDataFeatures.addAll(labelsAndFeatures.getFeatureVectors());
							}
							
							++splitCounter;
							
						}
					
						//Train the classifier
						classifier.train(trainingDataLabels, trainingDataFeatures);
						
						//Run predictions
						List<Integer> predictions = new ArrayList<Integer>();
						for (String testingDataVector : testingDataFeatures) {
							predictions.add(Integer.valueOf(classifier.predict(testingDataVector)));
						}
						
						//Get accuracy for this learning rate
						currentAccuracy = new ClassifierMetrics(testingDataLabels, predictions).getAccuracy();
						if (currentAccuracy > maximumAccuracy) {
							maximumAccuracy = currentAccuracy;
							bestLearningRate = learningRate;
							bestMyu = myu;
							bestNumberOfEpochs = epochs;
						}
					}
					
				}
			
			}
		}
		
		return this.new LearningRateAndMyu(bestLearningRate, bestMyu, bestNumberOfEpochs);

	}
	
	/**
	 * @param crossValidationFolds
	 * @param labels
	 * @param featureVectors
	 * @param numberOfFeatures
	 * @return learning rate from cross validation
	 */
	private LearningRateAndMyu getAggressiveMarginPerceptronLearningRateAndMyuByCrossValidation(int crossValidationFolds, List<Integer> labels, List<String> featureVectors, int numberOfFeatures, int[] epochValues) {
		
		double currentAccuracy = 0.0, maximumAccuracy = Double.MIN_VALUE, bestLearningRate = 0.0, bestMyu = 0.0;
		int bestNumberOfEpochs = 0;
		
		List<LabelsAndFeatures> crossValidationData = getCrossValidationData(crossValidationFolds, labels, featureVectors);
				
		for (int epochs : epochValues) {
			
			for (double myu : TEST_MYU_VALUES) {
	
				//Run k-fold cross validation
				for (int crossValidationCounter = 0; crossValidationCounter < crossValidationFolds; ++crossValidationCounter) {
					
					MarginPerceptron classifier = new AggressiveMarginPerceptron(false, epochs, numberOfFeatures, myu);
					
					//Load training and testing data
					List<Integer> trainingDataLabels = new ArrayList<Integer>();
					List<String> trainingDataFeatures = new ArrayList<String>();
					
					List<Integer> testingDataLabels = new ArrayList<Integer>();
					List<String> testingDataFeatures = new ArrayList<String>();
					
					int splitCounter = 0;
					for (LabelsAndFeatures labelsAndFeatures : crossValidationData) {
						
						if (splitCounter == crossValidationCounter) {
							testingDataLabels.addAll(labelsAndFeatures.getLabels());
							testingDataFeatures.addAll(labelsAndFeatures.getFeatureVectors());
						} else {
							trainingDataLabels.addAll(labelsAndFeatures.getLabels());
							trainingDataFeatures.addAll(labelsAndFeatures.getFeatureVectors());
						}
						
						++splitCounter;
						
					}
				
					//Train the classifier
					classifier.train(trainingDataLabels, trainingDataFeatures);
					
					//Run predictions
					List<Integer> predictions = new ArrayList<Integer>();
					for (String testingDataVector : testingDataFeatures) {
						predictions.add(Integer.valueOf(classifier.predict(testingDataVector)));
					}
					
					//Get accuracy for this learning rate
					currentAccuracy = new ClassifierMetrics(testingDataLabels, predictions).getAccuracy();
					if (currentAccuracy > maximumAccuracy) {
						maximumAccuracy = currentAccuracy;
						bestLearningRate = 0.0;
						bestMyu = myu;
						bestNumberOfEpochs = epochs;
					}
				}
					
			}
		}
		
		return this.new LearningRateAndMyu(bestLearningRate, bestMyu, bestNumberOfEpochs);

	}
	
	/**
	 * @param table2Features
	 * @return cleaned up feature vectors 
	 */
	private List<String> getCleanedTable2LabelsAndFeatures(List<String> table2Features) {
		
		List<String> cleanedFeatureVectors = new ArrayList<String>(table2Features.size());
		String[] columns = null, columnNumberAndValues = null;
		
		StringBuffer cleanedVector = null;
		for (String featureVector : table2Features) {
		
			cleanedVector = new StringBuffer();
			columns = featureVector.split(DataFileReader.WHITESPACE_REGEX);
			for (String column : columns) {
				
				columnNumberAndValues = column.split(PerceptronLinearClassifier.FEATURE_VALUE_SEPARATOR);
				try {
					cleanedVector.append(Integer.parseInt(columnNumberAndValues[0]) + 1);
					cleanedVector.append(PerceptronLinearClassifier.FEATURE_VALUE_SEPARATOR);
					cleanedVector.append(columnNumberAndValues[1]);
					cleanedVector.append(" ");
				} catch (NumberFormatException e) {
					System.exit(0);
				}
				
			}
			
			cleanedFeatureVectors.add(cleanedVector.toString());
		
		}
		
		return cleanedFeatureVectors;
		
	}
	
}
