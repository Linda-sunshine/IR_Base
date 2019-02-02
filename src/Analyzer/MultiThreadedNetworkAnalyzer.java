package Analyzer;

import opennlp.tools.util.InvalidFormatException;
import structures.*;
import utils.Utils;

import java.io.*;
import java.util.*;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 * The analyzer aims at the directional network information.
 */
public class MultiThreadedNetworkAnalyzer extends MultiThreadedLinkPredAnalyzer {

    HashMap<String, HashSet<String>> m_networkMap = new HashMap<String, HashSet<String>>();
    HashMap<String, Integer> m_CV4LinkMap;

    public MultiThreadedNetworkAnalyzer(String tokenModel, int classNo,
                                        String providedCV, int Ngram, int threshold, int numberOfCores, boolean b)
            throws IOException {
        super(tokenModel, classNo, providedCV, Ngram, threshold, numberOfCores, b);
    }

    // save one file for indexing users in later use (construct network for baselines)
    public void saveUserIds(String filename){
        try {
            PrintWriter writer = new PrintWriter(new File(filename));
            for(int i=0; i<m_users.size(); i++){
                writer.format("%s\t%d\n", m_users.get(i).getUserID(), i);
            }
            writer.close();
            System.out.format("Finish saving %d user ids.\n", m_users.size());
        } catch(IOException e){
            e.printStackTrace();
        }
    }
    // save the network for later use
    public void saveNetwork(String filename){
        try {
            PrintWriter writer = new PrintWriter(new File(filename));
            for(String uid: m_networkMap.keySet()){
                writer.write(uid + '\t');
                for(String it: m_networkMap.get(uid)){
                    writer.write(it + '\t');
                }
                writer.write('\n');
            }
            writer.close();
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    public void loadCV4Interactions(String filename){
        m_CV4LinkMap = new HashMap<>();
        try {
            if(m_userIDIndex==null)
                constructUserIDIndex();

            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;
            while((line = reader.readLine()) != null) {
                String[] strs = line.trim().split(",");
                String uId = strs[0];
                String vId = strs[1];
                int mask = Integer.valueOf(strs[2]);

                if(!m_userIDIndex.containsKey(uId))
                    System.out.format("[err]1st user %s not exits\n", uId);
                if(!m_userIDIndex.containsKey(vId))
                    System.out.format("[err]2nd user %s not exits\n", vId);
                if(m_userIDIndex.containsKey(uId) && m_userIDIndex.containsKey(vId)) {
                    m_CV4LinkMap.put(uId + " " + vId, mask);
                }
            }
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    // load the interactions, filter the users who are not in the user
    public void loadInteractions(String filename){
        m_networkMap.clear();
        try{
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;

            double avgFriendSize = 0;

            int count = 0;

            // load the interactions first
            while((line = reader.readLine()) != null){
                count++;
                String[] users = line.trim().split("\t");
                String uid = users[0];
                if(!m_userIDIndex.containsKey(users[0])){
                    System.err.format("[err]user %s does not exist in user set.\n", users[0]);
                    continue;
                }
                String[] interactions = Arrays.copyOfRange(users, 1, users.length);
                if(interactions.length == 0) continue;
                for(String in: interactions){
                    if(m_userIDIndex.containsKey(in)){
                        if(!m_networkMap.containsKey(uid))
                            m_networkMap.put(uid, new HashSet<String>());
                        if(!m_networkMap.containsKey(in))
                            m_networkMap.put(in, new HashSet<String>());
                        m_networkMap.get(uid).add(in);
                        m_networkMap.get(in).add(uid);
                    }
                }
//                System.out.println(count + "," + m_networkMap.size());
            }
            // set the friends for each user
            for(String ui: m_networkMap.keySet()){
                avgFriendSize += m_networkMap.get(ui).size();
                String[] frds = hashSet2Array(m_networkMap.get(ui));
                m_users.get(m_userIDIndex.get(ui)).setFriends(frds);
            }
            reader.close();
            avgFriendSize /= m_users.size();
            System.out.format("[Info]Total user size: %d, total doc size: %d, users with friends: %d, avg friend: " +
                    "%.3f.\n", m_users.size(), m_corpus.getCollection().size(), m_networkMap.size(), avgFriendSize);
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    protected String[] hashSet2Array(HashSet<String> strs){
        String[] arr = new String[strs.size()];
        int index = 0;
        for(String str: strs){
            arr[index++] = str;
        }
        return arr;
    }

    // shuffle the whole corpus and save the index information for later use
    public void saveCVIndex(int k, String filename){
        m_corpus.setMasks();
        m_corpus.shuffle(k);
        int[] masks = m_corpus.getMasks();
        ArrayList<_Doc> docs = m_corpus.getCollection();
        try{
            PrintWriter writer = new PrintWriter(new File(filename));
            for(int i=0; i<docs.size(); i++){
                _Review r = (_Review) docs.get(i);
                writer.write(String.format("%s,%d,%d\n", r.getUserID(), r.getID(), masks[i]));
            }
            writer.close();
            System.out.println("[Info]Finish writing cv index!");
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    @Override
    public void constructUserIDIndex(){
        m_userIDIndex = new HashMap<String, Integer>();
        for(int i=0; i<m_users.size(); i++){
            m_userIDIndex.put(m_users.get(i).getUserID(), i);
        }
    }

    // load cv index for all the documents
    public void loadCVIndex(String filename){
        try {
            if(m_userIDIndex==null)
                constructUserIDIndex();

            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;
            while((line = reader.readLine()) != null) {
                String[] strs = line.trim().split(",");
                String uId = strs[0];
                int id = Integer.valueOf(strs[1]);
                int mask = Integer.valueOf(strs[2]);

                if(!m_userIDIndex.containsKey(uId))
                    System.out.format("[error]No such user %s!\n", uId);
                else {
                    int uIndex = m_userIDIndex.get(uId);
                    if (uIndex > m_users.size())
                        System.out.println("Exceeds the array size!");
                    else {
                        if(m_users.get(m_userIDIndex.get(uId)).getReviewByID(id) != null)
                            m_users.get(m_userIDIndex.get(uId)).getReviewByID(id).setMask4CV(mask);
                    }
                }
            }
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    public _Doc getDocByUid(String uid, int index) {
        return m_users.get(m_userIDIndex.get(uid)).getReviewByID(index);
    }

    public void maskDocByCVIndex(int k){
        for(_User u : m_users){
            for(_Review r : u.getReviews()){
                int cvIdx = r.getMask4CV();
                if(cvIdx == k)
                    r.setType(_Review.rType.TEST);
                else
                    r.setType(_Review.rType.TRAIN);
            }
        }
    }

    public ArrayList<_Doc> getDocsByCVIndex(int k){
        ArrayList<_Doc> docs = new ArrayList<>();
        for(_User u : m_users){
            for(_Review r: u.getReviews()){
                int cvIdx = r.getMask4CV();
                if(cvIdx == k)
                    docs.add(r);
            }
        }
        return docs;
    }

    public void saveCV2Folds(String folder){
        try{
            for(_User u : m_users){
                for(_Review r : u.getReviews()){
                    int cvIdx = r.getMask4CV();
                    String cur_dir = String.format("%s/%d", folder, cvIdx);
                    new File(cur_dir).mkdirs();

                    File cur_file = new File(String.format("%s/%s", cur_dir, r.getUserID()));
                    if(!cur_file.exists()){
                        FileWriter file = new FileWriter(cur_file);
                        file.write(r.getUserID() + "\n");
                        file.flush();
                        file.close();
                    }

                    FileWriter file = new FileWriter(cur_file, true);
                    file.write(r.getItemID() + "\n");
                    file.write(r.getSource() + "\n");
                    file.write(r.getCategory() + "\n");
                    file.write(r.getYLabel() + "\n");
                    file.write(r.getTimeStamp() + "\n");
                    file.flush();
                    file.close();
                }
            }
        } catch (IOException e){
            e.printStackTrace();
        }
    }

    public void findUserWithMaxDocSize(){
        int max = -1;
        int count = 0;
        for(_User u: m_users){
            if(u.getReviewSize() > 1000) {
                System.out.println(u.getUserID());
                count++;
            }
            max = Math.max(max, u.getReviewSize());
        }
        System.out.println("Max doc size: " + max);
    }

    public void printDocs4Plane(String filename){
        try{
            PrintWriter writer = new PrintWriter(new File(filename));
            for(_User user: m_users){
                ArrayList<_SparseFeature[]> vectors = new ArrayList<>();
                for(_Review r: user.getReviews()){
                    vectors.add(r.getSparse());
                }
                _SparseFeature[] fvs = Utils.mergeSpVcts(vectors);
                for(_SparseFeature fv: fvs){
                    int index = fv.getIndex();
                    double val = fv.getValue();
                    for(int i=0; i<val; i++){
                        writer.write(index+" ");
                    }
                }
                writer.write("\n");
            }
            writer.close();
            System.out.println("Finish writing docs for PLANE!!");
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    /****We need to output some files for running baselines******/

//    // print out the network as an adjacency matrix with index in the friend file.
//    public void printNetwork4Plane(String filename){
//        try{
//            PrintWriter writer = new PrintWriter(new File(filename));
//
//            for(int i=0; i<m_users.size(); i++) {
//
//                String uid = m_userIds.get(i);
//                _User user = m_users.get(m_userIDIndex.get(uid));
//                if (user.getFriends() != null && user.getFriends().length > 0) {
//                    for (String frd : user.getFriends()) {
//                        writer.write(m_userMap4Network.get(frd) + "\t");
//                    }
//                    writer.write("\n");
//                }
//            }
//            writer.close();
//
//        } catch(IOException e){
//            e.printStackTrace();
//        }
//    }
//
//    // print out the network as an adjacency matrix with index in the friend file.
//    public void printAdjacencyMatrix(String filename){
//        try{
//            PrintWriter writer = new PrintWriter(new File(filename));
//            for(int i=0; i<m_userIds.size(); i++) {
//                String uid = m_userIds.get(i);
//                _User user = m_users.get(m_userIDIndex.get(uid));
//                if (user.getFriends() != null && user.getFriends().length > 0) {
//                    writer.write(i + "\t");
//                    for (String frd : user.getFriends()) {
//                        writer.write(m_userMap4Network.get(frd) + "\t");
//                    }
//                    writer.write("\n");
//                }
//            }
//            writer.close();
//            System.out.println("Finish saving adj matrix for ");
//        } catch(IOException e){
//            e.printStackTrace();
//        }
//    }

    public void printData4TADW(String filename){
        try{
            PrintWriter writer = new PrintWriter(new File(filename));
            for(_User user: m_users) {
                user.constructLRSparseVector();
                user.normalizeProfile();
                String uid = user.getUserID();
                for (_SparseFeature fv : user.getBoWProfile()) {
                    writer.write(String.format("%s\t%d\t%.3f\n", uid, fv.getIndex(), fv.getValue()));
                }
            }
            writer.close();
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    public void printData4CTR(BipartiteAnalyzer biAnalyzer, String dir, int testFold, int crossV, int groupIdx, boolean flag_cold){
        (new File(dir)).mkdirs();
        String flagstr = flag_cold?"_true":"_false";
        String flaggroup = groupIdx<0? "":String.format("_%d", groupIdx);
        String foldstr = crossV == 1? "":String.format("_%d", testFold);
        String corpusFile = String.format("%s/corpus%s%s.txt", dir, flagstr, foldstr);
        String trtFile = String.format("%s/train%s%s.txt", dir, flagstr, foldstr);
        String tstFile = String.format("%s/test%s%s%s.txt", dir, flagstr, foldstr, flaggroup);
        String userFile = String.format("%s/user%s%s.txt", dir, flagstr, foldstr);
        String userIdFile = String.format("%s/userID%s%s.txt", dir, flagstr, foldstr);
        String itemFile = String.format("%s/item%s%s.txt", dir, flagstr, foldstr);
        String itemIdFile = String.format("%s/itemID%s%s.txt", dir, flagstr, foldstr);

        try {
            ArrayList<_Doc> trainSet = new ArrayList<_Doc>();
            ArrayList<_Doc> testSet = new ArrayList<_Doc>();
            for (_User user : m_users) {
                for (_Review r : user.getReviews()) {
                    if(groupIdx<0) {
                        if (r.getMask4CV() == testFold)
                            testSet.add(r);
                        else
                            trainSet.add(r);
                    } else {
                        if (r.getMask4CV() == groupIdx)
                            testSet.add(r);
                        else if (r.getMask4CV() == 3)
                            trainSet.add(r);
                    }
                }
            }

            HashMap<Integer, ArrayList<Integer>> mapByUser = new HashMap<>();
            HashMap<Integer, ArrayList<Integer>> mapByItem = new HashMap<>();
            biAnalyzer.analyzeBipartite(m_corpus.getCollection(), "train");
            mapByUser = biAnalyzer.getMapByUser();
            mapByItem = biAnalyzer.getMapByItem();
            //print user -> items file
            PrintWriter writer = new PrintWriter(new File(userFile));
            for(int i = 0; i < biAnalyzer.getUsers().size(); i++){
                int len = mapByUser.get(i).size();
                writer.write(String.format("%d", len));
                for(int j = 0; j < len; j++) {
                    writer.write(String.format(" %d", mapByUser.get(i).get(j)));
                }
                writer.write("\n");
            }
            writer.close();

            writer = new PrintWriter(new File(userIdFile));
            List<_User> users = biAnalyzer.getUsers();
            for(int i = 0; i < users.size(); i++) {
                writer.write(String.format("%d\t%s\n", i, users.get(i).getUserID()));
            }
            writer.close();

            //print item -> users file
            writer = new PrintWriter(new File(itemFile));
            for(int i = 0; i < biAnalyzer.getItems().size(); i++){
                int len = mapByItem.get(i).size();
                writer.write(String.format("%d", len));
                for(int j = 0; j < len; j++) {
                    writer.write(String.format(" %d", mapByItem.get(i).get(j)));
                }
                writer.write("\n");
            }
            writer.close();

            writer = new PrintWriter(new File(itemIdFile));
            List<_Product> items = biAnalyzer.getItems();
            for(int i = 0; i < items.size(); i++) {
                writer.write(String.format("%d\t%s\n", i, items.get(i).getID()));
            }
            writer.close();

            //print train mult
            ArrayList<_Doc> itemBasedDocs = biAnalyzer.buildItemProfile(trainSet);
            saveDoc4LDA(trtFile, itemBasedDocs);

            //print test mult
            itemBasedDocs = biAnalyzer.buildItemProfile(testSet);
            saveDoc4LDA(tstFile, itemBasedDocs);

            //print all mult
            itemBasedDocs = biAnalyzer.buildItemProfile(m_corpus.getCollection());
            saveDoc4LDA(corpusFile, itemBasedDocs);

        } catch (IOException e){
            e.printStackTrace();
        }


    }

    public static void saveDoc4LDA(String filename, ArrayList<_Doc> docs) {
        if (filename==null || filename.isEmpty()) {
            System.out.println("Please specify the file name to save the vectors!");
            return;
        }

        try {
            BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename)));
            for(_Doc doc:docs) {
                writer.write(String.format("%d", doc.getSparse().length));
                for(_SparseFeature fv:doc.getSparse())
                    writer.write(String.format(" %d:%d", fv.getIndex(), (int) fv.getValue()));//index starts from 1
                writer.write("\n");
            }
            writer.flush();
            writer.close();

            System.out.format("[Info]%d feature vectors saved to %s\n", docs.size(), filename);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void printData4RTM_CVdoc(String dir, int testFold, int groupIdx, boolean flag_cold){
        (new File(dir)).mkdirs();
        String flagstr = flag_cold?"_coldstart":"";
        String flaggroup = groupIdx<0? "":String.format("_%d", groupIdx);
        String trtCorpusFile = String.format("%s/CVdoc%s_corpus_train_%d.txt", dir, flagstr, testFold);
        String trtLinkFile = String.format("%s/CVdoc%s_link_train_%d.txt", dir, flagstr, testFold);
        String userIdIdxFile = String.format("%s/CVdoc%s_userId_train_%d.txt", dir, flagstr, testFold);
        String tstCorpusFile = String.format("%s/CVdoc%s_corpus_test_%d%s.txt", dir, flagstr, testFold, flaggroup);
        String tstLinkFile = String.format("%s/CVdoc%s_link_test_train_%d%s.txt", dir, flagstr, testFold, flaggroup);
        String tsttstLinkFile = String.format("%s/CVdoc%s_link_test_test_%d%s.txt", dir, flagstr, testFold, flaggroup);

        try {
            //write train and test corpus
            PrintWriter writer_train = new PrintWriter(new File(trtCorpusFile));
            PrintWriter writer_test = new PrintWriter(new File(tstCorpusFile));
            HashMap<String, Integer> idx_train = new HashMap<>();
            HashMap<String, Integer> idx_test = new HashMap<>();
            int idx_train_size = 0, idx_test_size = 0;

            //construct sparse vector for this fold
            for (_User user : m_users) {
                _SparseFeature[] profile_train, profile_test;
                ArrayList<_SparseFeature[]> reviews_train = new ArrayList<_SparseFeature[]>();
                ArrayList<_SparseFeature[]> reviews_test = new ArrayList<_SparseFeature[]>();
                for (_Review r : user.getReviews()) {
                    if(groupIdx<0) {
                        if (r.getMask4CV() == testFold)
                            reviews_test.add(r.getSparse());
                        else
                            reviews_train.add(r.getSparse());
                    } else {
                        if (r.getMask4CV() == groupIdx)
                            reviews_test.add(r.getSparse());
                        else if (r.getMask4CV() == 3)
                            reviews_train.add(r.getSparse());
                    }
                }
                profile_train = Utils.MergeSpVcts(reviews_train);
                profile_test = Utils.MergeSpVcts(reviews_test);

                if(profile_train.length > 0){
                    writer_train.write(String.format("%d", calcTotalLength(profile_train)));
                    for(_SparseFeature fv : profile_train)
                        writer_train.write(String.format(" %d:%d", fv.getIndex(), (int) fv.getValue()));
                    writer_train.write("\n");

                    idx_train.put(user.getUserID(), idx_train_size++);
                }

                if(profile_test.length > 0){
                    writer_test.write(String.format("%d", calcTotalLength(profile_test)));
                    for(_SparseFeature fv : profile_test)
                        writer_test.write(String.format(" %d:%d", fv.getIndex(), (int) fv.getValue()));
                    writer_test.write("\n");

                    idx_test.put(user.getUserID(), idx_test_size++);
                }
            }
            writer_train.flush();
            writer_train.close();
            writer_test.flush();
            writer_test.close();

            System.out.format("[Info]%d training users saved to %s\n", idx_train_size, trtCorpusFile);
            System.out.format("[Info]%d testing users saved to %s\n", idx_test_size, tstCorpusFile);

            writer_train = new PrintWriter(new File(userIdIdxFile));
            for(String uId : idx_train.keySet()){
                writer_train.write(String.format("%d\t%s\n", idx_train.get(uId), uId));
            }
            writer_train.flush();
            writer_train.close();

            //write train test link
            writer_train = new PrintWriter(new File(trtLinkFile));
            writer_test = new PrintWriter(new File(tstLinkFile));
            int link_train_size=0, link_test_size=0;
            for(String uId : m_networkMap.keySet()){
                HashSet<String> friends = m_networkMap.get(uId);
                for(String vId : friends){
                    if(idx_train.containsKey(uId) && idx_train.containsKey(vId)){
                        writer_train.write(String.format("%d\t%d\n", idx_train.get(uId), idx_train.get(vId)));
                        link_train_size++;
                    }
                    if(idx_test.containsKey(uId) && idx_test.containsKey(vId)){
                        writer_test.write(String.format("%d\t%d\n", idx_test.get(uId), idx_test.get(vId)));
                        link_test_size++;
                    }
                }
            }
            writer_train.flush();
            writer_train.close();
            writer_test.flush();
            writer_test.close();
            (new File(tsttstLinkFile)).createNewFile();

            System.out.format("[Info]%d training links saved to %s\n", link_train_size, trtLinkFile);
            System.out.format("[Info]%d test links saved to %s\n", link_test_size, tstLinkFile);
        } catch (IOException e){
            e.printStackTrace();
        }
    }

    public void printData4RTM_CVlink(String dir, int testFold, boolean flag_cold){
        (new File(dir)).mkdirs();
        String flagstr = flag_cold?"_coldstart":"";
        String trtCorpusFile = String.format("%s/CVlink%s_corpus_train_%d.txt", dir, flagstr, testFold);
        String tstCorpusFile = String.format("%s/CVlink%s_corpus_test_%d.txt", dir, flagstr, testFold);
        String trtLinkFile = String.format("%s/CVlink%s_link_train_%d.txt", dir, flagstr, testFold);
        String tstLinkFile = String.format("%s/CVlink%s_link_test_train_%d.txt", dir, flagstr, testFold);
        String tsttstLinkFile = String.format("%s/CVlink%s_link_test_test_%d.txt", dir, flagstr, testFold);
        String userIdIdxFile = String.format("%s/CVlink%s_userId_train_%d.txt", dir, flagstr, testFold);

        try {
            PrintWriter writer_train = new PrintWriter(new File(trtCorpusFile));
            HashMap<String, Integer> idx_train = new HashMap<>();
            int idx_train_size = 0;
            for(_User user : m_users){
                _SparseFeature[] profile_train;
                ArrayList<_SparseFeature[]> reviews_train = new ArrayList<_SparseFeature[]>();
                for (_Review r : user.getReviews()) {
                    reviews_train.add(r.getSparse());
                }
                profile_train = Utils.MergeSpVcts(reviews_train);

                writer_train.write(String.format("%d", calcTotalLength(profile_train)));
                for(_SparseFeature fv : profile_train)
                    writer_train.write(String.format(" %d:%d", fv.getIndex(), (int) fv.getValue()));
                writer_train.write("\n");

                idx_train.put(user.getUserID(), idx_train_size++);
            }
            writer_train.flush();
            writer_train.close();
            (new File(tstCorpusFile)).createNewFile();

            System.out.format("[Info]CVIndex4Interation%s_fold_%d contains %d users.\n", flagstr,
                    testFold, idx_train.size());

            writer_train = new PrintWriter(new File(userIdIdxFile));
            for(String uId : idx_train.keySet()){
                writer_train.write(String.format("%d\t%s\n", idx_train.get(uId), uId));
            }
            writer_train.flush();
            writer_train.close();

            //write train test link
            writer_train = new PrintWriter(new File(trtLinkFile));
            int link_train_size=0;
            HashSet<String> invalid_user = new HashSet<>();
            for(String uId : m_networkMap.keySet()){
                HashSet<String> friends = m_networkMap.get(uId);
                for(String vId : friends){
                    if(idx_train.containsKey(uId) && idx_train.containsKey(vId)){
                        writer_train.write(String.format("%d\t%d\n", idx_train.get(uId), idx_train.get(vId)));
                        link_train_size++;
                    } else {
                        if (!idx_train.containsKey(uId))
                            invalid_user.add(uId);
                        else if(!idx_train.containsKey(vId))
                            invalid_user.add(vId);
                    }
                }
            }
            writer_train.flush();
            writer_train.close();
            (new File(tsttstLinkFile)).createNewFile();
            (new File(tstLinkFile)).createNewFile();

            System.out.format("[Info]%d training links (%d-%d users) saved to %s\n",
                    link_train_size, idx_train.size(), invalid_user.size(), trtLinkFile);

            System.out.print("invalide user that not in corpus:\n");
            for(String uId : invalid_user){
                System.out.format("-- %s\n", uId);
            }
        } catch (IOException e){
            e.printStackTrace();
        }
    }

    private int calcTotalLength(_SparseFeature[] x_sparse) {
        int length = 0;
        for(_SparseFeature fv : x_sparse)
            length += fv.getValue();
        return length;
    }

    public void printData4HFT(String dir, String source, String mode) throws IOException{
        String outFile = String.format("%s/%s_data.tsv", dir, mode);
        (new File(outFile)).getParentFile().mkdirs();

        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outFile)));
        //calculate the average upvotes for stackoverflow user
        if(source.equals("StackOverflow") && mode.equals("CVlink")){
            for(_User u : m_users){
                int total_upvotes = 0;
                for(_Review r : u.getReviews())
                    total_upvotes += r.getYLabel();
                float ave_upvotes = u.getReviewSize() > 0?(float)total_upvotes/u.getReviewSize():0;
                for(_Review r : u.getReviews())
                    r.setYLabel(r.getYLabel()>=ave_upvotes?1:0);
            }
        }

        int writenum = 0;
        HashSet<String> valid_users = new HashSet<>();
        for(_Doc doc : m_corpus.getCollection()) {
            //userID itemID rating time docLength words
            _Review r = (_Review) doc;
//            if(mode.equals("CVdoc") && r.getItemID().equals("-1"))
//                continue;
            writenum++;
            valid_users.add(r.getUserID());

            String userID = r.getUserID();
            String itemID = r.getItemID();
            int rate = r.getYLabel();
            writer.write(String.format("%s\t%s\t%d\t0", userID, itemID, rate));

            writer.write(String.format("\t%d", doc.getTotalDocLength()));
            for(_SparseFeature fv:doc.getSparse()) {
                int count = (int) fv.getValue();
                String word = m_corpus.getFeature(fv.getIndex());
                for(int i = 0; i < count; i++){
                    writer.write(String.format("\t%s", word));//index starts from 1
                }
            }
            writer.write("\n");
        }
        writer.close();

        System.out.format("[Info]%d in %d rates (from %d-%d users) saved to %s\n",
                writenum, m_corpus.getCollection().size(), valid_users.size(), m_users.size(), outFile);
    }


}
