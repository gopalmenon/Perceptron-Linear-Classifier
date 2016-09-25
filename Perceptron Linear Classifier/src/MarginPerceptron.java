
/**
 * Margin Perceptron
 *
 */
public class MarginPerceptron extends PerceptronLinearClassifier {

	private double myu;
	
	//Constructor
	public MarginPerceptron(boolean useZeroWeights, int numberOfEpochs, double learningRate, int numberOfFeatures, double myu) {
		
		super(useZeroWeights, numberOfEpochs, learningRate, numberOfFeatures);
		this.myu = myu;
		
	}

	/**
	 * @param dotProductOfWeightsAndFeatures
	 * @return true if weights and bias need to be updated
	 */
	@Override
	protected boolean weightUpdateRequired(double dotProductOfWeightsAndFeatures, int labelValue) {
		
		if (dotProductOfWeightsAndFeatures * (double) labelValue <= this.myu) {
			return true;
		} else {
			return false;
		}
		
	}

}
