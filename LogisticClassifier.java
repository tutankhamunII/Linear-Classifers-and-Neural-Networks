package learn.lc.core;

import learn.math.util.VectorOps;

public class LogisticClassifier extends LinearClassifier {
	
	public LogisticClassifier(double[] weights) {
		super(weights);
	}
	
	public LogisticClassifier(int ninputs) {
		super(ninputs);
	}
	
	/**
	 * A LogisticClassifier uses the logistic update rule
	 * (AIMA Eq. 18.8): w_i \leftarrow w_i+\alpha(y-h_w(x)) \times h_w(x)(1-h_w(x)) \times x_i 
	 */
	public void update(double[] x, double y, double alpha) {
        // calculate the predicted output
        double predicted = eval(x);

        // update weights based on the logistic regression learning rule: wi  = wi + a * (y-predicted) * predicted * (1-predicted) * xi
        for (int i = 0; i < weights.length; i++) {
            weights[i] += alpha * (y - predicted) * predicted * (1 - predicted) * x[i];
        }
	}
	
	/**
	 * A LogisticClassifier uses a 0/1 sigmoid threshold at z=0.
	 */
	public double threshold(double z) {
		//sigmoid function
		return 1 / (1 + Math.exp(-1*z));
	}

}
