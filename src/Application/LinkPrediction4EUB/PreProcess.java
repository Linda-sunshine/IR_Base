package Application.LinkPrediction4EUB;

import Analyzer.MultiThreadedNetworkAnalyzer;
import opennlp.tools.util.InvalidFormatException;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 */
public class PreProcess {

    int m_userSize = 0;
    HashMap<String, HashSet<String>> m_networkMap = new HashMap<>();
    ArrayList<String> m_userIds = new ArrayList<String>();
    HashMap<String, Integer> m_idIndexMap = new HashMap<>();

    // load user ids for later use
    public void loadUserIds(String idFile) {
        try {
            m_userIds = new ArrayList<>();
            File file = new File(idFile);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;

            while ((line = reader.readLine()) != null) {
                String uid = line.trim();
                m_idIndexMap.put(uid, m_userIds.size());
                m_userIds.add(uid);
            }
            m_userSize = m_userIds.size();
            System.out.format("Finish loading %d user ids from %s.\n", m_userIds.size(), idFile);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // load the interactions, filter the users who are not in the user
    public void loadInteractions(String filename) {
        try {
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;

            // load the interactions first
            while ((line = reader.readLine()) != null) {
                String[] users = line.trim().split("\\s+");
                String uid = users[0];
                if (!m_idIndexMap.containsKey(users[0])) {
                    System.err.println("The user does not exist in user set!");
                    continue;
                }
                String[] interactions = Arrays.copyOfRange(users, 1, users.length);
                if (interactions.length == 0) continue;
                for (String in : interactions) {
                    if (in.equals(uid)) continue;
                    if (m_idIndexMap.containsKey(in)) {
                        if (!m_networkMap.containsKey(uid))
                            m_networkMap.put(uid, new HashSet<String>());
                        if (!m_networkMap.containsKey(in))
                            m_networkMap.put(in, new HashSet<String>());
                        m_networkMap.get(uid).add(in);
                        m_networkMap.get(in).add(uid);
                    }
                }
            }

            int missing = 0;
            for (String ui : m_networkMap.keySet()) {
                for (String frd : m_networkMap.get(ui)) {
                    if (!m_networkMap.containsKey(frd))
                        missing++;
                    if (!m_networkMap.get(frd).contains(ui))
                        System.out.println("Asymmetric!!");
                }
            }
            if (missing > 0)
                System.out.println("[error]Some edges are not in the set: " + missing);
            System.out.println(m_networkMap.size());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void writeInteractions4DeepWalk(String filename){
        try{
            PrintWriter writer = new PrintWriter(new File(filename));
            for(String ui: m_networkMap.keySet()) {
                int i = m_idIndexMap.get(ui);
                for (String uj : m_networkMap.get(ui)) {
                    int j = m_idIndexMap.get(uj);
                    writer.format("%d\t%d\n", i, j);
                }
            }
            writer.close();
            System.out.format("Finish writing %d users interactions for deepwalk.\n", m_networkMap.size());
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    public void transferDWEmbedding(String input, String output, int dim){
        try {
            PrintWriter writer = new PrintWriter(new File(output));
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(input),
                    "UTF-8"));
            int count = 0;
            // first line: num of user * dim
            String line = reader.readLine();
            HashMap<String, String> embeddings = new HashMap<>();
            while ((line = reader.readLine()) != null) {
                String[] strs = line.split("\\s+");
                int id = Integer.valueOf(strs[0]);
                String uid = m_userIds.get(id);
                embeddings.put(uid, getEmbedding(strs));
                count++;
            }
            reader.close();
            String empty = "";
            for(int i=0; i<dim; i++){
                empty = empty + "0\t";
            }
            empty += "";
            writer.format("%d\t%d\n", m_userIds.size(), dim);
            for(String uid: m_userIds){
                if(embeddings.containsKey(uid)) {
                    writer.format("%s\t%s\n", uid, embeddings.get(uid));
                } else{
                    writer.format("%s\t%s\n", uid, empty);
                }
            }
            writer.close();
            System.out.format("Finish transferring %d user embeddings from %s to %s.\n", count, input, output);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public String getEmbedding(String[] strs){
        String str = "";
        for(int i=1; i<strs.length; i++)
            str += strs[i] + "\t";
        return str;
    }
    //In the main function, we want to input the data and do adaptation
    public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException {


        int k = 0;
        String dataset = "FB"; // "StackOverflow", "YelpNew"
        String prefix = "./data/RoleEmbedding";
        String idPrefix = "/Users/lin"; // "/Users/lin", "/home/lin"
        String idFile = String.format("%s/DataWWW2019/UserEmbedding/%s_userids.txt", idPrefix, dataset);

        String cvIndexFile4Interaction = String.format("%s/%sCVIndex4Interaction_fold_%d_train.txt", prefix, dataset, k);

        // write out the data for deepwalk
//        String dwTrainFile = String.format("%s/%s_LINE_fold_%d.txt", prefix, dataset, k);
//        PreProcess p = new PreProcess();
//        p.loadUserIds(idFile);
//        p.loadInteractions(cvIndexFile4Interaction);
//        p.writeInteractions4DeepWalk(dwTrainFile);

        for(String model: new String[]{"DW_len_8" }){
            for(int dim: new int[]{10}) {
                PreProcess p = new PreProcess();
                p.loadUserIds(idFile);
                p.loadInteractions(cvIndexFile4Interaction);

                // transfer the output of deepwalk/line for embedding
//                String input = String.format("%s/Dropbox/output/%s_%s_user_embedding_dim_%d_fold_%d.txt", idPrefix,
//                        dataset, model, dim, k);
                String input = String.format("%s/Documents/Lin\'sWorkSpace/deepwalk-master/data/%s_%s_dim_%d_fold_%d.txt", idPrefix,
                        dataset, model, dim, k);
                String output = String.format("%s/DataWWW2019/UserEmbedding/%s_%s_embedding_dim_%d_fold_%d.txt",
                        idPrefix, dataset, model, dim, k);
                p.transferDWEmbedding(input, output, dim);
            }
        }
    }
}
