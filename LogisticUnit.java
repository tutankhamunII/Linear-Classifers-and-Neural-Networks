package learn.nn.core;
import java.util.List;
import learn.math.util.VectorOps;

/**
 * A LogisticUnit is a Unit that uses a sigmoid
 * activation function.
 */
public class LogisticUnit extends NeuronUnit {
	
	/**
	 * The activation function for a LogisticUnit is a 0-1 sigmoid
	 * centered at z=0: 1/(1+e^(-z)). (AIMA Fig 18.7)
	 */
	@Override
	public double activation(double z) {
		return 1 / (1+Math.pow(Math.E, -1*z));
	}
	
	/**
	 * Derivative of the activation function for a LogisticUnit.
	 * For g(z)=1/(1+e^(-z)), g'(z)=g(z)*(1-g(z)) (AIMA p. 727).
	 * @see https://calculus.subwiki.org/wiki/Logistic_function#First_derivative
	 */
	public double activationPrime(double z) {
		double y = activation(z);
		return y * (1.0 - y);
	}

	/**
	 * Update this unit's weights using the logistic regression
	 * gradient descent learning rule (AIMA Eq 18.8).
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
			incoming.get(i).weight += alpha * (y-predicted) * predicted *(1-predicted)* input[i];
		}
	}
	
}
