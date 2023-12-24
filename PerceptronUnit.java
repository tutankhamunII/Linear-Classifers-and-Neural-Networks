package learn.nn.core;
import java.util.List;

import learn.math.util.VectorOps;
/**
 * A PerceptronUnit is a Unit that uses a hard threshold
 * activation function.
 */
public class PerceptronUnit extends NeuronUnit {
	
	/**
	 * The activation function for a Perceptron is a hard 0/1 threshold
	 * at z=0. (AIMA Fig 18.7)
	 */
	@Override
	public double activation(double z) {
		return (z >= 0) ? 1.0 : 0.0;
	}
	@Override
	public double activationPrime(double z){
		return 0.0;
	}
	/**
	 * Update this unit's weights using the Perceptron learning
	 * rule (AIMA Eq 18.7).
	 * Remember: If there are n input attributes in vector x,
	 * then there are n+1 weights including the bias weight w_0. 
	 */
	@Override
	public void update(double[] x, double y, double alpha) {
		List<Connection> incoming = this.incomingConnections;
		double w[] = new double[incoming.size()]; //weight vector
		for(int i =0; i < incoming.size(); i++){
			w[i] = incoming.get(i).weight;
		}
		double[] input = new double[x.length+1];
		input[0]= 1.0; //bias
		for(int i =1; i < input.length;i++){
			input[i] = x[i];
		}
		double predicted = this.activation(VectorOps.dot(w, input)); //calculate weighted sum and apply activation
		for(int i = 0; i < input.length; i++){
			incoming.get(i).weight += alpha * (y-predicted) * input[i];
		}
	}
}
