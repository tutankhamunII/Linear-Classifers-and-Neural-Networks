
import java.io.*;
import java.util.*;
import learn.nn.core.Connection;
import learn.nn.core.Example;
import learn.nn.core.InputUnit;
import learn.nn.core.LogisticUnit;
import learn.nn.core.MultiLayerFeedForwardNeuralNetwork;
import learn.nn.core.Unit;



public class Iris extends MultiLayerFeedForwardNeuralNetwork{
    static int layers_num = 3;
    static int input_num = 4;
    static int hidden = 7;
    static int output_num = 3;
    public Iris() {
		super(new Unit[layers_num][]);
		// Input units
		this.layers[0] = new InputUnit[input_num];
		for (int i=0; i < input_num; i++) {
			this.layers[0][i] = new InputUnit();
		}
		// Hidden units: each connected to all input units
		this.layers[1] = new LogisticUnit[hidden];
		for (int j=0; j < hidden; j++) {
			this.layers[1][j] = new LogisticUnit();
			for (int i=0; i < input_num; i++) {
				new Connection(this.layers[0][i], this.layers[1][j]);
			}
		}
		// Output units: each connected all hidden units
		this.layers[2] = new LogisticUnit[output_num];
		for (int j=0; j < output_num; j++) {
			this.layers[2][j] = new LogisticUnit();
			for (int i=0; i < hidden; i++) {
				new Connection(this.layers[1][i], this.layers[2][j]);
			}
		}
	}
    public static void main(String[] args) throws IOException{
        int epochs = Integer.parseInt(args[0]);
        double alpha = Double.parseDouble(args[1]);
        String file = args[2];
        String output_file = args[3];
        String output_file2 = args[4];

        ArrayList<Example> examples=new ArrayList<Example>();
        Scanner scnr = new Scanner(new File(file));
        while(scnr.hasNextLine()){
            String[] input_string = scnr.nextLine().split(",");
            double[] input_double = new double[input_string.length-1];
            for(int i = 0; i < input_double.length; i++){
                input_double[i] = Double.parseDouble(input_string[i]);
            }
            String label = input_string[input_string.length-1];
            double[] output = new double[3];
            if(label.equals("Iris-setosa")){
                output[0] = 1.0;
            }
            else if(label.equals("Iris-versicolor")){
                output[1] = 1.0;
            }
            else if(label.equals("Iris-virginica")){
                output[2] = 1.0;
            }
            Example ex = new Example(input_double, output);
            examples.add(ex);
        }
        Iris nn = new Iris();
        //graph (a)
        try (PrintWriter writer = new PrintWriter(new FileWriter(output_file))) {
            for (epochs=0; epochs <= 400; epochs+=50) {
                nn.train(examples, epochs, alpha);
                double accuracy = nn.test(examples);
                double error = (1-accuracy)*100;
                writer.println(epochs + "\t" + error);
            }
        } 
        catch (IOException e) {
            e.printStackTrace();
        }
        //graph (b)
        try (PrintWriter writer = new PrintWriter(new FileWriter(output_file2))) {
            Collections.shuffle(examples);
            for (int size = 1; size < 100; size++) {
                nn.train(examples.subList(0, size), epochs, alpha);
                double accuracy = nn.test(examples.subList(size-1, examples.size()));
                writer.println(size + "\t" + accuracy);
                Collections.shuffle(examples);
            }
        } 
        catch (IOException e) {
            e.printStackTrace();
        }
    } 
}
