package learn.lc.core;

import learn.math.util.VectorOps;

public class PerceptronClassifier extends LinearClassifier {
	
	public PerceptronClassifier(double[] weights) {
		super(weights);
	}
	
	public PerceptronClassifier(int ninputs) {
		super(ninputs);
	}
	
	/**
	 * A PerceptronClassifier uses the perceptron learning rule
	 * (AIMA Eq. 18.7): w_i \leftarrow w_i+\alpha(y-h_w(x)) \times x_i 
	 */
	public void update(double[] x, double y, double alpha) {
		//calculate the predicted output
		double predicted = eval(x);
		//update weights based on the perceptron learning rule: wi = wi + a * (y-predicted) * xi
        for (int i = 0; i < weights.length; i++) {
            weights[i] += alpha * (y - predicted) * x[i];
        }
	}
	
	/**
	 * A PerceptronClassifier uses a hard 0/1 threshold.
	 */
	public double threshold(double z) {
		//hard threshold function for perceptron
		return (z > 0) ? 1 : 0;	}
	
}
