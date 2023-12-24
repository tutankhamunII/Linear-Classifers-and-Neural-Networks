import java.util.*;
import learn.lc.core.Example;
import learn.lc.core.PerceptronClassifier;
import learn.lc.core.LogisticClassifier;
import learn.lc.core.DecayingLearningRateSchedule;
import java.io.*;
public class Main {
    public static void main(String args[]){
        String filename = args[0];
        String type = args[1];
        int nsteps = Integer.parseInt(args[2]);
        double alpha = Double.parseDouble(args[3]);
        String output_file = args[4];
        File input_file = new File(filename);
        ArrayList<Example> examples=new ArrayList<Example>();
        try {
            Scanner scnr = new Scanner(input_file);
            while(scnr.hasNextLine()){
                String[] input_string = scnr.nextLine().split(",");
                double[] input_double = new double[input_string.length];
                input_double[0] = 1.0; //set the first element to 1.0 (bias)
                for(int i = 1; i < input_double.length; i++){
                    input_double[i] = Double.parseDouble(input_string[i - 1]);
                }
                double output = Double.parseDouble(input_string[input_string.length - 1]);//set last element as output
                Example ex = new Example(input_double, output);
                examples.add(ex);
            }
            
        } catch (Exception e) {
            System.out.println("File not found");
            e.printStackTrace();
        }
        DecayingLearningRateSchedule alpha_decay= new DecayingLearningRateSchedule();
        ArrayList<Double> graph_arr = new ArrayList<Double>();
        if(type.equals("perceptron")){
            PerceptronClassifier p = new PerceptronClassifier(examples.get(0).inputs.length);
            //use alpha_decay
            if(alpha == 0.0){
                p.train(examples, nsteps, alpha_decay);
            }
            else{ //use constant alpha from user input
                p.train(examples, nsteps, alpha);
            }
            graph_arr = p.train_arr;
        }else if(type.equals("logistic")){
            LogisticClassifier l = new LogisticClassifier(examples.get(0).inputs.length);
            if(alpha == 0){
                l.train(examples, nsteps, alpha_decay);
            }else{
                l.train(examples, nsteps, alpha);
            }
            graph_arr = l.train_arr;
        } else{
            System.out.println("Invalid classifier, type 'perceptron' or 'logistic'");
        }
        try (PrintWriter writer = new PrintWriter(new FileWriter(output_file))) {
            for (int i = 1; i <= nsteps; i++) {
                writer.println(i + "\t" + graph_arr.get(i-1));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
