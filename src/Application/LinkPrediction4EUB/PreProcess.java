package Application.LinkPrediction4EUB;

import Analyzer.MultiThreadedNetworkAnalyzer;
import jdk.nashorn.internal.parser.JSONParser;
import opennlp.tools.util.InvalidFormatException;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import org.json.*;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 */
public class PreProcess {

    int m_userSize = 0;
    HashMap<String, HashSet<String>> m_networkMap = new HashMap<>();
    ArrayList<String> m_userIds = new ArrayList<String>();
    // key: user_id, value: index
    HashMap<String, Integer> m_idIndexMap = new HashMap<>();

    // splitter cannot tolerate random index set, thus we need to transfer the user id into continuous index range
    HashMap<String, Integer> m_idIndexMap4Splitter = new HashMap<>();

    // key: index used in splitter, value: user_id
    HashMap<Integer, String> m_indexIdMap4Splitter = new HashMap<>();

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

    // load user ids for later use
    public void loadSplitterUserIds(String filename) {
        try {
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;

            int count = 0;
            while ((line = reader.readLine()) != null) {
                String uid = line.trim();
                m_indexIdMap4Splitter.put(count, uid);
                m_idIndexMap4Splitter.put(uid, count);
                count++;
            }
            System.out.format("Finish loading %d user ids for splitter from %s.\n", count, filename);
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
                if (!m_idIndexMap.containsKey(uid)) {
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

            // sanity checks for missing links
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

    // we write out the interactions in the form of their index defined in the userid file.
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


    public void writeInteractions4SPLITTER(String interactionFile, String idFile){
        try{
            int index = 0;
            for(String uid: m_networkMap.keySet()){
                m_idIndexMap4Splitter.put(uid, index++);
            }
            PrintWriter writer = new PrintWriter(new File(interactionFile));
            for(String ui: m_networkMap.keySet()) {
                int uiIdx = m_idIndexMap4Splitter.get(ui);
                for (String uj : m_networkMap.get(ui)) {
                    int ujIdx = m_idIndexMap4Splitter.get(uj);
                    writer.format("%d,%d\n", uiIdx, ujIdx);
                }
            }
            writer.close();
            System.out.format("Finish writing %d users interactions for SPLITTER into %s.\n", m_networkMap.size(), interactionFile);
            writer = new PrintWriter(new File(idFile));
            for(String ui: m_networkMap.keySet()){
                writer.write(ui+"\n");
            }
            writer.close();
            System.out.format("Finish writing %d users ids for SPLITTER into %s.\n", m_networkMap.size(), idFile);
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

    HashMap<Integer, HashSet<Integer>> m_user2Persona = new HashMap<>();
    HashMap<Integer, Integer> m_persona2User = new HashMap<>();
    public void loadPersonas(String filename){

        try {

            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename),
                    "UTF-8"));
            // first line: num of user * dim
            String line = reader.readLine();
            JSONObject obj = new JSONObject(line);
            int size = obj.length();
            for(int i=0; i<size; i++){
                int personaIndex = i;
                String key = String.format("%d", i);
                int userIndex = obj.getInt(key);

                if(!m_user2Persona.containsKey(userIndex)){
                    m_user2Persona.put(userIndex, new HashSet<>());
                }
                m_user2Persona.get(userIndex).add(personaIndex);
                m_persona2User.put(personaIndex, userIndex);
            }
            System.out.format("Finish loading %d mapping pairs from %s.\n", size, filename);
        } catch (IOException e)  {
            e.printStackTrace();
        }
    }

    public void transferSPLITTEREmbeddings(String input, String output, int dim){
        try {
            PrintWriter writer = new PrintWriter(new File(output));
            writer.format("%d\t%d\n", m_userIds.size(), dim);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(input),
                    "UTF-8"));
            // first line: num of user * dim
            String line = reader.readLine();
            int count = 0;
            while ((line = reader.readLine()) != null) {
                String[] strs = line.split(",");
                // the persona index of the embedding
                int personaIndex = Double.valueOf(strs[0]).intValue();
                // the user index in trainning splitter
                int userIndex = m_persona2User.get(personaIndex);
                String userId = m_indexIdMap4Splitter.get(userIndex);
                writer.format("%s\t%s\n", userId, getEmbedding(strs));
                count++;
            }
            reader.close();
            writer.close();
            System.out.format("Finish transferring %d user embeddings from %s to %s.\n", count, input, output);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    //In the main function, we want to input the data and do adaptation
    public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException {


        int k = 0, dim = 10, nuWalks = 10, walkLen = 40;
        String dataset = "YelpNew"; // "StackOverflow", "YelpNew"
        String prefix = "./data/RoleEmbedding";
        String idPrefix = "/Users/lin"; // "/Users/lin", "/home/lin"
        String idFile = String.format("%s/DataWWW2019/UserEmbedding/%s_userids.txt", idPrefix, dataset);

        String cvIndexFile4Interaction = String.format("%s/%sCVIndex4Interaction_fold_%d_train.txt", prefix, dataset, k);

//        // ****** write out the data for deepwalk *****
//        String dwTrainFile = String.format("%s/%s_DW_fold_%d.txt", prefix, dataset, k);
//        PreProcess p = new PreProcess();
//        p.loadUserIds(idFile);
//        p.loadInteractions(cvIndexFile4Interaction);
//        p.writeInteractions4DeepWalk(dwTrainFile);

//        //****** write out the data for splitter *****
//        PreProcess p = new PreProcess();
//        p.loadUserIds(idFile);
//        p.loadInteractions(cvIndexFile4Interaction);
        String splitterInteractionFile = String.format("./data/SPLITTER/%s_SPLITTER_fold_%d.txt", dataset, k);
        String splitterIdFile = String.format("./data/SPLITTER/%s_SPLITTER_index_fold_%d.txt", dataset, k);
//        p.writeInteractions4SPLITTER(splitterInteractionFile, splitterIdFile);

//        // ***** transfer the output of deepwalk/line for embedding *****
//        // since the learned user embeddings are indexed with number, we need to transfer them to user_ids for link prediction
//        String dwModel = String.format("DW_len=%d_nu=%d", walkLen, nuWalks);
//        PreProcess p = new PreProcess();
//        p.loadUserIds(idFile);
//        p.loadInteractions(cvIndexFile4Interaction);
//        String input = String.format("%s/Documents/Lin\'sWorkSpace/deepwalk-master-lin-refined/output/%s_%s_embedding_dim_%d_fold_%d.txt", idPrefix,dataset, dwModel, dim, k);
//        String output = String.format("%s/DataWWW2019/UserEmbedding/%s_%s_embedding_dim_%d_fold_%d.txt", idPrefix, dataset, dwModel, dim, k);
//        p.transferDWEmbedding(input, output, dim);


        // ***** transfer the output of splitter for embedding *****
        PreProcess p = new PreProcess();
        p.loadUserIds(idFile);
        p.loadSplitterUserIds(splitterIdFile);
        String embeddingFile = String.format("./data/SPLITTER/YelpNew_SPLITTER_embedding_dim_10_fold_0.txt");
        String personaFile = String.format("./data/SPLITTER/YelpNew_SPLITTER_personas_dim_10_fold_0.json");

        String output = String.format("%s/DataWWW2019/UserEmbedding/%s_SPLITTER_embedding_dim_%d_fold_%d.txt", idPrefix, dataset, dim, k);
        p.loadPersonas(personaFile);
        p.transferSPLITTEREmbeddings(embeddingFile, output, dim);
    }
}
