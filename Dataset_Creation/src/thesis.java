import java.io.*;
import java.util.ArrayList;

/**
 * Created by Shusmoy on 12/20/2017.
 */
public class thesis {

    ArrayList<word> words;
    ArrayList<line> lines;
    ArrayList<pair> pairs;
    public int x=1;
    ArrayList<String> appIds;
    public  void data(String file)

    {

        try {
            FileInputStream inputStream = new FileInputStream(file);
            InputStreamReader reader = new InputStreamReader(inputStream, "UTF-16");
            int character;
            String sw="";
            String sl="";
            words= new ArrayList<>();
            lines = new ArrayList<>();
            pairs= new ArrayList<>();
            appIds = new ArrayList<>();
            while ((character = reader.read()) != -1) {
                char s = (char)character;
                //System.out.print((char) character);
                if(s=='।')
                {
                    word w= new word(sw);
                    words.add(w);
                    word w1= new word("।");
                    words.add(w1);
                    sw="";
                    line line=new line(sl,"।");
                    lines.add(line);
                    sl="";
                    continue;
                }
                else if(s==';')
                {
                        word w= new word(sw);
                        words.add(w);
                        word w1= new word(";");
                        words.add(w1);
                        sw="";
                    line line=new line(sl,";");
                    lines.add(line);
                    sl="";
                        continue;
                }
                sw=sw+s;
                sl=sl+s;
                if(s==' ')
                {
                    //if(!word.contains(sw))
                    word w= new word(sw);
                    words.add(w);
                    sw="";
                }
                if (s=='\n')
                {
                    line line=new line(sl,"\n");
                    lines.add(line);
                    sl="";
                    //System.out.println(sy+"\n");
                }

            }
            reader.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
        setcount();
        //System.out.println("Word"+ " "+ "count"+ " "+"index");
        //for(int i=0;i<words.size();i++) {
            //System.out.println(words.get(i).getW()+" "+words.get(i).getC()+" "+words.get(i).getIndex());
            //System.out.println(lines.get(i).getL());

        //}

    }
    public void slidingwindow(int n)
    {
       for(int i=0;i<words.size();i++)
       {

           for(int j=i+1;j<n+i;j++)
           {
               if((words.get(i).getW().equals("।"))||(words.get(i).getW().equals(";")))
               {
                   i++;
                   if(i>=words.size()) break;
               }
               else if((words.get(j).getW().equals("।"))||(words.get(j).getW().equals(";")))
               {

                   j++;
                   if(j>=words.size()) break;
               }
               else {
                   pair p = new pair(words.get(i).getIndex(), words.get(j).getIndex(), words.get(i).getW(), words.get(j).getW());
                   pairs.add(p);
               }
           }
       }
    }
    public void setcount()
    {
        int f=0;
        for(int i=0;i<words.size();i++)
        {
            f=0;
            for(int j=0;j<words.size();j++)
            {
                if(words.get(i).contain(words.get(j).getW()))
                {
                    if(i>j) {
                        words.get(i).setIndex(words.get(j).getIndex());
                        words.get(i).setC(words.get(j).getC());
                        f = 1;
                        break;
                    }
                    else
                    {
                        words.get(i).c++;
                    }
                }


            }
            if(f!=1)
            {
                words.get(i).setIndex(x);
                x++;
            }
        }
    }
    public void pairprint()
    {
        for(int i=0;i<pairs.size();i++)
            System.out.println(pairs.get(i).getP1()+","+pairs.get(i).getP2());
        System.out.println(pairs.size());
    }
    public void  makepair()
    {
        for(int i=0;i<words.size()-1;i++)
        {
            if((words.get(i).getW().equals("।")||(words.get(i+1).getW().equals("।")))||(words.get(i).getW().equals(";"))||(words.get(i+1).getW().equals(";")))
            {
                i++;
            }
            else
            {
                pair p= new pair(words.get(i).getIndex(),words.get(i+1).getIndex(),words.get(i).getW(),words.get(i+1).getW());
                pairs.add(p);
            }
        }

    }
    public String makeonehot(int y)
    {
        String s="[";
        for(int i=1;i<=x;i++)
        {
            if(i==y) s=s+"1";
            else s=s+"0";
            if(i!=x) s=s+",";
        }
        s=s+"]\n";
        return s;
    }
    public void datawrite() throws IOException {
        FileOutputStream outputStream = new FileOutputStream("data.txt");
        OutputStreamWriter writer = new OutputStreamWriter(outputStream, "UTF-16");
        int c=0;
        for (int i=0;i<pairs.size();i++)
        {
           // writer.write("Pair "+(i+1)+"\n");
            String pair= "Pair"+(i+1)+"\n";
            String p1= makeonehot(pairs.get(i).getP1());
            String p2= makeonehot(pairs.get(i).getP2());
            //appIds.add(pair);
            appIds.add(p1);
            appIds.add(p2);
            //writer.write(p1);
            //writer.write(p2);
            //writer.flush();
            c++;


        }
        System.out.println(appIds.size()+" "+c);
        DumpPickle dumpPickle= new DumpPickle();
        dumpPickle.dumpHashsetToPickle(appIds);
        //PrintWriter writer = new PrintWriter("the-file-name.txt", "UTF-16");
        //writer.println("The first line");
        //writer.println("The second line");
        writer.close();
    }
    public static void main(String[] args) throws IOException {
        String f= "H:\\Soroswati Puja\\data.txt";
        thesis t= new thesis();
        t.data(f);
        t.slidingwindow(5);
        t.pairprint();
        //t.makepair();
        t.datawrite();
    }
}
