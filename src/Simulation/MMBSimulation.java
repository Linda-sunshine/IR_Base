package Simulation;

import org.apache.commons.math3.distribution.BinomialDistribution;
import utils.Utils;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

public class MMBSimulation {

    protected double[][] m_B;
    protected int m_dim;
    protected int m_userSize;
    protected double[][] m_userMixtures;
    protected HashMap<Integer, HashSet<Integer>> m_network;
    protected BinomialDistribution m_bernoulli;

    public MMBSimulation(int dim, int userSize){
        m_dim = dim;
        m_userSize = userSize;
    }

    public void simulate(){
        randomGenerateB();
        randomGenerateUserMixture();
        generateNetwork();
    }

    public void saveEverything(String dir){
        saveAffinityMatrix(dir);
        saveUserMixtures(dir);
        saveNetwork(dir);
        saveEdgeAssignment(dir);
    }

    public void saveAffinityMatrix(String dir){
        try{
            PrintWriter writer = new PrintWriter(new File(dir+"_B.txt"), "UTF-8");
            writer.format("%d\t%d\n", m_dim, m_dim);
            for(int i=0; i<m_B.length; i++){
                writer.write(i+"\t");
                for(int j=0; j<m_B.length; j++){
                    writer.format("%.4f\t", m_B[i][j]);
                }
                writer.write("\n");
            }
            writer.close();
            System.out.println("[Info]Finish writing affinity matrix!");
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    public void saveUserMixtures(String dir){
        try{
            PrintWriter writer = new PrintWriter(new File(dir+"_user_mixture.txt"), "UTF-8");
            writer.format("%d\t%d\n", m_userSize, m_dim);
            for(int i=0; i<m_userSize; i++){
                writer.write(i+"\t");
                for(int j=0; j<m_dim; j++){
                    writer.format("%.4f\t", m_userMixtures[i][j]);
                }
                writer.write("\n");
            }
            writer.close();
            System.out.println("[Info]Finish writing user mixtures!");
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    public void saveNetwork(String dir){
        try{
            PrintWriter writer = new PrintWriter(new File(dir+"_interactions.txt"), "UTF-8");
            for(int id: m_network.keySet()){
                writer.write(id+"\t");
                HashSet<Integer> frds = m_network.get(id);
                for(int frdIdx: frds){
                    writer.format("%d\t", frdIdx);
                }
                writer.write("\n");
            }
            writer.close();
            System.out.format("[Info]Finish writing %d users' network data!", m_network.size());
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    public void saveEdgeAssignment(String dir){
        try{
            PrintWriter writer = new PrintWriter(new File(dir+"_edgeAssignment.txt"), "UTF-8");
            writer.write("ui\tuj\tindicator_i\tindicator_j\n");
            for(String s: m_edgeAssignment){
                writer.write(s + "\n");
            }
            writer.close();
            System.out.format("[Info]Finish writing %d edge assignment information!", m_edgeAssignment.size());
        } catch(IOException e){
            e.printStackTrace();
        }
    }


    protected void randomGenerateB(){
        m_B = new double[m_dim][m_dim];
        for(int i=0; i<m_B.length; i++){
            for(int j=i; j<m_B[0].length; j++) {
                m_B[i][j] = Math.random();
                m_B[j][i] = m_B[i][j];
            }
        }
        for(int i=0; i<m_B.length; i++) {
            Utils.normalizeByWeight(m_B[i], 20);
        }
    }

    protected void randomGenerateUserMixture(){
        m_userMixtures = new double[m_userSize][m_dim];
        for(int i=0; i<m_userSize; i++){
            for(int j=0; j<m_dim; j++){
                m_userMixtures[i][j] = Math.random();
            }
            Utils.normalize(m_userMixtures[i]);
        }
    }

    ArrayList<String> m_edgeAssignment = new ArrayList<String>();
    protected void generateNetwork(){
        m_network = new HashMap<>();
        for(int i=0; i<m_userSize; i++){
            for(int j=i+1; j<m_userSize; j++){
                int indicatorI = sampleIndicator(m_userMixtures[i]);
                int indicatorJ = sampleIndicator(m_userMixtures[j]);
                double affinity = m_B[indicatorI][indicatorJ];

                m_bernoulli = new BinomialDistribution(1, affinity);
                if(m_bernoulli.sample() == 1){
                    if(!m_network.containsKey(i)){
                        m_network.put(i, new HashSet<>());
                    }
                    if(!m_network.containsKey(j)){
                        m_network.put(j, new HashSet<>());
                    }
                    m_network.get(i).add(j);
                    m_network.get(j).add(i);
                    String edgeOne = String.format("%d\t%d\t%d\t%d", i, j, indicatorI, indicatorJ);
                    String edgeTwo = String.format("%d\t%d\t%d\t%d", j, i, indicatorJ, indicatorI);
                    m_edgeAssignment.add(edgeOne);
                    m_edgeAssignment.add(edgeTwo);
                }
            }
        }
        double avg = 0;
        for(int idx: m_network.keySet()){
            avg += m_network.get(idx).size();
        }
        avg /= m_network.size();
        System.out.format("[Info]Finish generating social connections!! Avg number of connections: %.2f.\n", avg);
    }

    // for each user, construct the
    public void constructValidSet(){
        HashSet<Integer> trainFrds = new HashSet<>();
        HashSet<Integer> testFrds = new HashSet<>();
        HashSet<Integer> trainNonFrds = new HashSet<>();
        HashSet<Integer> testNonFrds = new HashSet<>();
        try {
            PrintWriter trainFrdWriter = new PrintWriter(new File("./data/simulation/SimulationCVIndex4Interaction_fold_0_train.txt"));
            PrintWriter testFrdWriter = new PrintWriter(new File("./data/simulation/SimulationCVIndex4Interaction_fold_0_test.txt"));
            PrintWriter trainNonFrdWriter = new PrintWriter(new File("./data/simulation/SimulationCVIndex4NonInteraction_fold_0_train.txt"));
            PrintWriter testNonFrdWriter = new PrintWriter(new File("./data/simulation/SimulationCVIndex4NonInteraction_fold_0_test.txt"));

            for (int idx : m_network.keySet()) {
                trainFrds.clear();
                testFrds.clear();
                testNonFrds.clear();

                HashSet<Integer> frds = m_network.get(idx);
                for (int frdIdx : frds) {
                    if (Math.random() > 0.8) {
                        testFrds.add(frdIdx);
                    } else {
                        trainFrds.add(frdIdx);
                    }
                }
                // sample non friends for testing
                while (testNonFrds.size() < trainFrds.size() * 2) {
                    int iterIdx = (int) (Math.random() * m_network.size());
                    // if the user is not the current user &&  current user's friends
                    if (iterIdx != idx && !frds.contains(iterIdx)) {
                        testNonFrds.add(iterIdx);
                    }
                    iterIdx++;
                }
                // sample non friends for training
                while (trainNonFrds.size() < trainFrds.size() * 2) {
                    int iterIdx = (int) (Math.random() * m_network.size());
                    // if the user is not the current user &&  current user's friends
                    if (iterIdx != idx && !frds.contains(iterIdx) && !testNonFrds.contains(iterIdx)) {
                        trainNonFrds.add(iterIdx);
                    }
                    iterIdx++;
                }
                if(trainFrds.size() > 0){
                    trainFrdWriter.write(idx + "\t");
                    // write out the friends information
                    for(int i: trainFrds){
                        trainFrdWriter.write(i + "\t");
                    }
                    trainFrdWriter.write("\n");
                }
                if(testFrds.size() > 0){
                    testFrdWriter.write(idx + "\t");
                    for(int i: testFrds){
                        testFrdWriter.write(i + "\t");
                    }
                    testFrdWriter.write("\n");
                }
                if(trainNonFrds.size() > 0){
                    trainNonFrdWriter.write(idx + "\t");
                    for(int i: trainNonFrds){
                        trainNonFrdWriter.write(i + "\t");
                    }
                    trainNonFrdWriter.write("\n");
                }
                if(testNonFrds.size() > 0){
                    testNonFrdWriter.write(idx + "\t");
                    for(int i: testNonFrds){
                        testNonFrdWriter.write(i + "\t");
                    }
                    testNonFrdWriter.write("\n");
                }
            }
            trainFrdWriter.close();
            testFrdWriter.close();
            trainNonFrdWriter.close();
            testNonFrdWriter.close();
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    protected int sampleIndicator(double[] ui){
        double rdm = Math.random();
        int left = 0, right = ui.length-1;
        double[] sum = new double[ui.length];
        for(int i=0; i<ui.length; i++){
            sum[i] = (i == 0 ? 0: sum[i-1]) + ui[i];
        }
        while(left < right){
            int mid = (left + right) / 2;
            if(sum[mid] == rdm)
                return mid;
            else if(rdm > sum[mid]){
                left = mid + 1;
            } else
                right = mid;
        }
        return left;
    }

    double[][] m_trueB;
    double[][] m_estB;

    public double compareTwoBs(String trueBFile, String estBFile){
        m_trueB = loadB(trueBFile);
        m_estB = loadB(estBFile);
        double sim = 0;
        for(int i=0; i<m_trueB.length; i++){
            double tmp = Utils.cosine(m_trueB[i], m_estB[i]);
            sim += tmp;
            System.out.println(tmp);
        }
        sim /= m_trueB.length;
        System.out.format("The similarity is %.2f.\n", sim);
        return sim;
    }

    public double[][] loadB(String filename){
        try {
            // load beta for the whole corpus first
            File userFile = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(userFile),
                    "UTF-8"));
            int count = 0;
            String line = reader.readLine(); // access the first line to get the dims
            String[] strs = line.split("\t");
            int nuRoles = Integer.valueOf(strs[0]);
            int dim = Integer.valueOf(strs[1]);
            double[][] B = new double[nuRoles][dim];
            while ((line = reader.readLine()) != null){
                strs = line.trim().split("\t");
                int roleIdx = Integer.valueOf(strs[0]);
                double[] embedding = new double[strs.length - 1];
                for(int i=1; i<strs.length; i++)
                    embedding[i-1] = Double.valueOf(strs[i]);
                B[roleIdx] = embedding;
            }
            System.out.format("[Info]Finish loading %d role embeddings from %s\n", nuRoles, filename);
            return B;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static void main(String[] args) {
        int dim = 10, userSize = 1000;
        MMBSimulation sim = new MMBSimulation(dim, userSize);
//        sim.simulate();
//        sim.saveEverything(String.format("./data/simulation/Simulation_dim_%d_user_%d", dim, userSize));
//        sim.constructValidSet();

        String trueBFile = String.format("./data/Simulation_dim_10_user_1000_B.txt");
        String estBFile = String.format("./data/Simulation_role_l1_embedding_alpha_0.0010_step_size_0.0010_iter_100_order_1_nuOfRoles_10_dim_10_fold_0.txt");
        sim.compareTwoBs(trueBFile, estBFile);

    }
}
