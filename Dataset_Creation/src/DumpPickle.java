import org.python.core.PyFile;
import org.python.core.PyList;
import org.python.core.PyString;
import org.python.modules.cPickle;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashSet;







public class DumpPickle {

    public static void main(String[] args) {

        HashSet<String> appIds = new HashSet<String>();

        appIds.add("1234321432");

        appIds.add("1234321433");

        appIds.add("xsydfsflkfds");

        DumpPickle afpd = new DumpPickle();

        //afpd.dumpHashsetToPickle(appIds);

    }



    public void dumpHashsetToPickle(ArrayList<String> pIds) {

        File f = new File("data.pkl");

        OutputStream fs = null;

        try {

            fs = new FileOutputStream(f);

        } catch (FileNotFoundException e) {

            e.printStackTrace();

            return;

        }

        PyFile pyF = new PyFile(fs);

        PyList pyIdList = new PyList();

        for (String id : pIds) {

            PyString pyStr = new PyString(id);

            pyIdList.add(pyStr);

        }

        cPickle.dump(pyIdList, pyF);

        pyF.flush();

        pyF.close();

    }



}