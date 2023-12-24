import java.io.*;
import java.util.*;
import learn.nn.core.Example;
import learn.nn.core.MultiLayerFeedForwardNeuralNetwork;

public class MNIST extends MultiLayerFeedForwardNeuralNetwork {
	
	static final int num_input = 28*28;
	static final int hidden = 300;
	static final int num_out = 10;
	
	public MNIST() {
		super(num_input, hidden, num_out);
	}
	
	public static void main(String[] args) throws IOException {
		int epochs = 200;
		double alpha = 0.10;
		MNIST nn = new MNIST();
        String in_1 = args[0];
        String in_2 = args[1];
        String in_3 = args[2];
        String in_4 = args[3];
        String out_file = args[4];
        String out_file2 = args[5];
        String mutual_path = "learn/nn/examples/";
        System.out.println("reading...");
        List<Example> training = Reader.read(mutual_path + in_1, mutual_path + in_2);
		List<Example> testing = Reader.read(mutual_path + in_3, mutual_path + in_4);
        System.out.println("done reading...");
        Collections.shuffle(training);
        Collections.shuffle(testing);
        List<Example> training_subList = training.subList(0, 1000);
        List<Example> testing_subList = testing.subList(0, 1000);

        //graph (a)
        try (PrintWriter writer = new PrintWriter(new FileWriter(out_file))) {
            System.out.println("starting training");
            for (epochs=0; epochs <= 10; epochs+=1) {
                nn.train(training_subList, epochs, alpha);
                System.out.println("training with epoch: " + epochs +" is complete");
                double accuracy = nn.test(testing_subList);
                double error = (1-accuracy)*100;
                writer.println(epochs + "\t" + error);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        //graph (b)
        System.out.println("graph b");
        try (PrintWriter writer = new PrintWriter(new FileWriter(out_file2))) {
            for (int size=1; size < 50; size++) {
                System.out.println("training with set of size " + size);
                nn.train(training_subList.subList(0, size), 2, alpha);
                double accuracy = nn.test(testing_subList);
                writer.println(size + "\t" + accuracy);
                Collections.shuffle(training_subList);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
	}
	
}