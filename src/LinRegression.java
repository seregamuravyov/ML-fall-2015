import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by Администратор on 11.09.2015.
 */
public class LinRegression {

    private Instances dataset;

    public void loadDataset(String filename) throws IOException {
        File file = new File(filename);

        FastVector attributes = new FastVector();
        attributes.addElement(new Attribute("square"));
        attributes.addElement(new Attribute("roomNum"));
        attributes.addElement(new Attribute("price"));

        dataset = new Instances("flatPrice", attributes, 50);

        List<ArrayList<Integer>> data = new ArrayList<ArrayList<Integer>>();
        BufferedReader bf = new BufferedReader(new FileReader(file));

        String text = null;
        while ((text = bf.readLine()) != null) {
            String [] tmpSplit = text.split(",");
            List<Integer> currInstance = new ArrayList<>();
            Instance curr = new Instance(3);
            for (int i = 0; i < tmpSplit.length; i++) {
                curr.setValue(i, Double.parseDouble(tmpSplit[i]));
            }
            dataset.add(curr);
        }
    }

    public Instances getDataset(){
        return dataset;
    }

    public static void main(String [] args) throws Exception {
        LinRegression lr = new LinRegression();
        lr.loadDataset("prices.txt");

        Instances data = lr.getDataset();
        data.setClassIndex(data.numAttributes() - 1);

        LinearRegression regression = new LinearRegression();
        //System.out.print(data.lastInstance().numAttributes());

        regression.buildClassifier(data);
        System.out.println(regression.toString());

        ///regression.

        Instance myHouse = new Instance(3);

        myHouse.setValue(0, 2004);
        myHouse.setValue(1, 4);
        myHouse.setMissing(2);

        //Instance myHouse = data.lastInstance();
        double price = regression.classifyInstance(myHouse);
        System.out.println("My house ("+myHouse+"): "+price);

        //Regression regression = new Regression()

        //weka.estimators.NormalEstimator

    }
}
