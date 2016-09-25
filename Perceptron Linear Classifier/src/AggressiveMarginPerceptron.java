import java.util.List;


/**
 * Aggressive perceptron with margin
 *
 */
public class AggressiveMarginPerceptron extends MarginPerceptron {
	
	//Constructor
	public AggressiveMarginPerceptron(boolean useZeroWeights, int numberOfEpochs, double learningRate, int numberOfFeatures, double myu) {
		
		super(useZeroWeights, numberOfEpochs, learningRate, numberOfFeatures, myu);
		
	}
	
	/**
	 * Update the weights and bias 
	 * @param label
	 * @param featureValues
	 */
	@Override
	protected void updateWeightsAndBias(int label, List<Integer> featureValues, double dotProductOfWeightsAndFeatures) {
		
		int weightsCounter = 0;
		double learningRate = getLearningRate(label, featureValues, dotProductOfWeightsAndFeatures);
		
		for (Double weight : this.weights) {
			
			if (weightsCounter == 0) {
				//Update the bias
				this.weights.set(weightsCounter, Double.valueOf(weight.doubleValue() +  (learningRate * (double) label)));
			} else {
				//Update the weight
				this.weights.set(weightsCounter, Double.valueOf(weight.doubleValue() +  (learningRate * (double) label * featureValues.get(weightsCounter).doubleValue())));
			}
			++weightsCounter;
		}
		
	}
	
	/**
	 * @param label
	 * @param featureValues
	 * @param dotProductOfWeightsAndFeatures
	 * @return learning rate eta
	 */
	private double getLearningRate(int label, List<Integer> featureValues, double dotProductOfWeightsAndFeatures) {
		
		return (this.myu - (double) label * dotProductOfWeightsAndFeatures) / getDotOfTransposeProductWithVector(featureValues);
		
	}
	
	/**
	 * @param featureValues
	 * @return dot product of input vector transpose with the input vector
	 */
	private int getDotOfTransposeProductWithVector(List<Integer> featureValues) {
		
		int dotProductWithTranspose = 0;
		
		for (Integer featureValue : featureValues) {
			dotProductWithTranspose += featureValue * featureValue;
		}
		
		
		return dotProductWithTranspose;
		
	}
	
}
