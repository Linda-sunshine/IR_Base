package myMains;

import opennlp.tools.util.InvalidFormatException;

import java.io.*;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 */
public class MyProcessMain {

    public static HashMap<String, ArrayList<String>[]> loadFoldData(String filename, int kFold){

        HashMap<String, ArrayList<String>[]> m_userFoldMap = new HashMap<>();
        try {
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;
            // load the interactions first
            while ((line = reader.readLine()) != null) {
                String[] strs = line.trim().split(",");
                String ui = strs[0], uj = strs[1];
                int k = Integer.valueOf(strs[2]);
                if (!m_userFoldMap.containsKey(ui)) {
                    ArrayList<String>[] tmp = new ArrayList[kFold];
                    for(int kf=0; kf<kFold; kf++){
                        tmp[kf] = new ArrayList<String>();
                    }
                    m_userFoldMap.put(ui, tmp);
                }
                m_userFoldMap.get(ui)[k].add(uj);
            }
            System.out.format("%d users' fold information is loaded!\n", m_userFoldMap.size());
        } catch (IOException e){
            e.printStackTrace();
        }
        return m_userFoldMap;
    }

    public static void printTrainTestData(HashMap<String, ArrayList<String>[]> userFoldMap, String dir, int kFold){

        int trainCount = 0, testCount = 0;
        // the currnet fold number
        for(int k=0; k<kFold; k++){
            trainCount = 0;
            testCount = 0;
            try{
                String trainFile = String.format("%sCVIndex4Interaction_fold_%d_train.txt", dir, k);
                String testFile = String.format("%sCVIndex4Interaction_fold_%d_test.txt", dir, k);

                PrintWriter trainWriter = new PrintWriter(new File(trainFile));
                PrintWriter testWriter = new PrintWriter(new File(testFile));
                for(String uid: userFoldMap.keySet()) {
                    ArrayList<String> trainIds = new ArrayList<>();
                    for (int l = 0; l < kFold; l++) {
                        // test data
                        if (l == k) {
                            if (userFoldMap.get(uid)[k].size() == 0) {
                                continue;
                            }
                            testWriter.write(uid + "\t");
                            for (String uj : userFoldMap.get(uid)[k]) {
                                testWriter.write(uj + "\t");
                                testCount++;
                            }
                            testWriter.write("\n");
                        } else {
                            // collect train data
                            trainIds.addAll(userFoldMap.get(uid)[l]);
                        }
                    }
                    if (trainIds.size() > 0) {
                        trainWriter.write(uid + "\t");
                        for (String uj : trainIds) {
                            trainWriter.write(uj + "\t");
                            trainCount++;
                        }
                        trainWriter.write("\n");
                    }
                }
                trainWriter.close();
                testWriter.close();
                System.out.print(String.format("[fold-%d]train edge size: %d, test edge size: %d\n", k, trainCount, testCount));
            } catch(IOException e){
                e.printStackTrace();
            }
        }
        System.out.println("Finish writing all fold information!");
    }

    public static void printTestData(HashMap<String, ArrayList<String>[]> userFoldMap, String dir, int kFold, int time){

        int testCount = 0;
        // the currnet fold number
        for(int k=0; k<kFold; k++){
            testCount = 0;
            try{
                String testFile = String.format("%sCVIndex4NonInteraction_time_%d_fold_%d.txt", dir, time, k);

                PrintWriter testWriter = new PrintWriter(new File(testFile));
                for(String uid: userFoldMap.keySet()) {
                    for (int l = 0; l < kFold; l++) {
                        ArrayList<String> trainIds = new ArrayList<>();
                        // test data
                        if (l == k) {
                            if (userFoldMap.get(uid)[k].size() == 0) {
                                continue;
                            }
                            testWriter.write(uid + "\t");
                            for (String uj : userFoldMap.get(uid)[k]) {
                                testWriter.write(uj + "\t");
                                testCount++;
                            }
                            testWriter.write("\n");
                        }
                    }
                }
                testWriter.close();
                System.out.print(String.format("[fold-%d]Test edge size: %d\n", k, testCount));
            } catch(IOException e){
                e.printStackTrace();
            }
        }
        System.out.println("Finish writing all fold information!");
    }

    public static HashSet<String> loadOriginalCV(String filename){
        HashSet<String> features = new HashSet<>();
        try {
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line = reader.readLine(); // skip the first line

            // load the interactions first
            while ((line = reader.readLine()) != null) {
                String[] strs = line.trim().split(",");
                features.add(strs[1]);
            }
            reader.close();
        } catch(IOException e){
            e.printStackTrace();
        }
        return features;
    }

    public static HashSet<String> loadDFFeatures(String filename){
        HashSet<String> features = new HashSet<>();
        try {
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line; // skip the first line

            // load the interactions first
            while ((line = reader.readLine()) != null) {
                if(line.startsWith("#"))
                    continue;
                features.add(line.trim());
            }
            reader.close();
        } catch(IOException e){
            e.printStackTrace();
        }
        return features;
    }

    public static void writeFeatures(String filename, HashSet<String> fs){
        try{
            PrintWriter writer = new PrintWriter(new File(filename));
            for(String f: fs){
                writer.write(f+"\n");
            }
            writer.close();
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    public static void checkOverlapping(HashSet<String> fs1, HashSet<String> fs2){
        int count = 0;
        for(String fs: fs1){
            if(fs2.contains(fs)){
                System.out.println(fs);
                count++;
            }
        }
        System.out.format("%d features in common!", count);
    }

    public static void main(String[] args){
        int kFold = 5;
        String dataset = "YelpNew";
        String prefix = "./data/CoLinAdapt";

        String orgCV = String.format("%s/%s/SelectedVocab.csv", prefix, dataset);
        String df3kCV = String.format("%s/%s/SelectedVocab_DF_5k.txt", prefix, dataset);
        String ig6kCV = String.format("%s/%s/SelectedVocab_DF_5k.txt", prefix, dataset);

//        String orgCVtxt = String.format("%s/%s/SelectedVocab.txt", prefix, dataset);
//        HashSet<String> fs1 = loadOriginalCV(orgCV);
//        writeFeatures(orgCVtxt, fs1);

//        HashSet<String> fs1 = loadDFFeatures(df3kCV);
//        HashSet<String> fs2 = loadDFFeatures(ig6kCV);
//        checkOverlapping(fs1, fs2);


        String cvIndexFile4Interaction = String.format("%s/%s/%sCVIndex4Interaction.txt", prefix, dataset, dataset);
        String dir = String.format("%s/%s/%s", prefix, dataset, dataset);
        printTrainTestData(loadFoldData(cvIndexFile4Interaction, kFold), dir, kFold);

        for(int time: new int[]{2, 3, 4}) {
            String cvIndexFile4NonInteraction = String.format("%s/%s/%sCVIndex4NonInteraction_time_%d.txt", prefix, dataset, dataset, time);
            printTestData(loadFoldData(cvIndexFile4NonInteraction, kFold), dir, kFold, time);
        }
    }
}
