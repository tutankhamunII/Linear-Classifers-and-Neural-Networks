package learn.lc.core;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import learn.math.util.VectorOps;

abstract public class LinearClassifier {
	
	public double[] weights;
	public ArrayList<Double> train_arr; //keeps track of accuracy throughout training 
	public LinearClassifier(double[] weights) {
		this.weights = weights;
		this.train_arr = new ArrayList<>();
	}
	
	public LinearClassifier(int ninputs) {
		this(new double[ninputs]);
	}
	
	/**
	 * Update the weights of this LinearClassifer using the given
	 * inputs/output example and learning rate alpha.
	 */
	abstract public void update(double[] x, double y, double alpha);

	/**
	 * Threshold the given value using this LinearClassifier's
	 * threshold function.
	 */
	abstract public double threshold(double z);

	/**
	 * Evaluate the given input vector using this LinearClassifier
	 * and return the output value.
	 * This value is: Threshold(w \cdot x)
	 */
	public double eval(double[] x) {
		return threshold(VectorOps.dot(this.weights, x));
	}
	
	/**
	 * Train this LinearClassifier on the given Examples for the
	 * given number of steps, using given learning rate schedule.
	 * ``Typically the learning rule is applied one example at a time,
	 * choosing examples at random (as in stochastic gradient descent).''
	 * See AIMA p. 724.
	 */
	public void train(List<Example> examples, int nsteps, LearningRateSchedule schedule) {
		Random random = new Random();
		int n = examples.size();
		for (int i=1; i <= nsteps; i++) {
			int j = random.nextInt(n);
			Example ex = examples.get(j);
			this.update(ex.inputs, ex.output, schedule.alpha(i));
			this.trainingReport(examples, i,  nsteps);
		}
	}

	/**
	 * Train this LinearClassifier on the given Examples for the
	 * given number of steps, using given constant learning rate.
	 */
	public void train(List<Example> examples, int nsteps, double constant_alpha) {
		train(examples, nsteps, new LearningRateSchedule() {
			public double alpha(int t) { return constant_alpha; }
		});
	}
	
	/**
	 * This method is called after each weight update during training.
	 * Subclasses can override it to gather statistics or update displays.
	 */
	protected void trainingReport(List<Example> examples, int stepnum, int nsteps) {
		train_arr.add(accuracy(examples));
		//System.out.println(stepnum + "\t" + accuracy(examples));
	}
	
	/**
	 * Return the squared error per example (Mean Squared Error) for this
	 * LinearClassifier on the given Examples.
	 * The Mean Squared Error is the total L_2 loss divided by the number
	 * of samples.
	 */
	public double squaredErrorPerSample(List<Example> examples) {
		double sum = 0.0;
		for (Example ex : examples) {
			double result = eval(ex.inputs);
			double error = ex.output - result;
			sum += error*error;
		}
		return sum / examples.size();
	}

	/**
	 * Return the proportion of the given Examples that are classified
	 * correctly by this LinearClassifier.
	 * This is probably only meaningful for classifiers that use
	 * a hard threshold. Use with care.
	 */
	public double accuracy(List<Example> examples) {
		int ncorrect = 0;
		for (Example ex : examples) {
			double result = eval(ex.inputs);
			// System.out.println(result +" "+ex.output);
			if(result > 0.0 && result < 1.0){ //logistic
				if(result > 0.5){
					result = 1.0;
				}
				else{
					result = 0.0;
				}
			}
			if (result == ex.output) {
				ncorrect += 1;
			}
		}
		return (double)ncorrect / examples.size();
	}

}
