
import java.io.*;
import java.util.*;
import learn.nn.core.Example;

public class Reader {

	static class InputStream {

		FileInputStream stream;
		int count;

		protected InputStream(String filename, int num) throws IOException {
			this.stream = new FileInputStream(filename);
			int n = readInt32();
			if (n != num) {
				throw new IOException("	INVALID: " + n);
			}
			this.count = readInt32();
		}

		protected int readByte() throws IOException {
			int n = stream.read();
			if (n == -1) {
				throw new EOFException();
			}
			return n;
		}

		protected int readInt32() throws IOException {
			long n = 0;
			for (int i=0; i < 4; i++) {
				n = n*256 + readByte();
			}
			return (int)n;
		}
	}

	static class ImageInputStream extends InputStream {
		static int num = 2051;
		protected int nrows;
		protected int ncols;
		public ImageInputStream(String filename) throws IOException {
			super(filename, num);
			this.nrows = readInt32();
			this.ncols = readInt32();
		}

		public int[] nextImage() throws IOException {
			int[] data = new int[nrows*ncols]; // Java bytes are signed
			// could just do fis.read(data) ...
			for (int i=0; i < nrows * ncols; i++) {
				data[i] = readByte();
			}
			return data;
		}

	}

	static class LabelInputStream extends InputStream {
		private static final int MAGIC = 2049;
		public LabelInputStream(String filename) throws IOException {
			super(filename, MAGIC);
		}
		public int nextLabel() throws IOException {
			return readByte();
		}
	}

	static public List<Example> read(String imageFilename, String labelFilename) throws IOException {
		ImageInputStream istream = new ImageInputStream(imageFilename);
		LabelInputStream lstream = new LabelInputStream(labelFilename);
		if (istream.count != lstream.count) {
			throw new IOException("UNEQUAL: " + istream.count + " " + lstream.count);
		}
		int n = istream.count;
		List<Example> examples = new ArrayList<Example>(n);
		for (int i=0; i < n; i++) {
			int[] imageData = istream.nextImage();
			int label = lstream.nextLabel();
			double[] inputs = new double[imageData.length];
			for (int j=0; j < imageData.length; j++) {
				inputs[j] = (double)imageData[j];
			}
			double[] outputs = new double[100];
			outputs[label] = 1.0;
			examples.add(new Example(inputs, outputs));
		}
		return examples;
	}
}