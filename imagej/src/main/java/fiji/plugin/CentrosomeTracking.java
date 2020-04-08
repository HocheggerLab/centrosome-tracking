package main.java.fiji.plugin;

import static fiji.plugin.trackmate.tracking.TrackerKeys.DEFAULT_ALLOW_GAP_CLOSING;
import static fiji.plugin.trackmate.tracking.TrackerKeys.DEFAULT_ALLOW_TRACK_MERGING;
import static fiji.plugin.trackmate.tracking.TrackerKeys.DEFAULT_ALLOW_TRACK_SPLITTING;
import static fiji.plugin.trackmate.tracking.TrackerKeys.DEFAULT_ALTERNATIVE_LINKING_COST_FACTOR;
import static fiji.plugin.trackmate.tracking.TrackerKeys.DEFAULT_BLOCKING_VALUE;
import static fiji.plugin.trackmate.tracking.TrackerKeys.DEFAULT_CUTOFF_PERCENTILE;
import static fiji.plugin.trackmate.tracking.TrackerKeys.DEFAULT_GAP_CLOSING_FEATURE_PENALTIES;
import static fiji.plugin.trackmate.tracking.TrackerKeys.DEFAULT_GAP_CLOSING_MAX_DISTANCE;
import static fiji.plugin.trackmate.tracking.TrackerKeys.DEFAULT_GAP_CLOSING_MAX_FRAME_GAP;
import static fiji.plugin.trackmate.tracking.TrackerKeys.DEFAULT_LINKING_FEATURE_PENALTIES;
import static fiji.plugin.trackmate.tracking.TrackerKeys.DEFAULT_LINKING_MAX_DISTANCE;
import static fiji.plugin.trackmate.tracking.TrackerKeys.DEFAULT_MERGING_FEATURE_PENALTIES;
import static fiji.plugin.trackmate.tracking.TrackerKeys.DEFAULT_MERGING_MAX_DISTANCE;
import static fiji.plugin.trackmate.tracking.TrackerKeys.DEFAULT_SPLITTING_FEATURE_PENALTIES;
import static fiji.plugin.trackmate.tracking.TrackerKeys.DEFAULT_SPLITTING_MAX_DISTANCE;
import static fiji.plugin.trackmate.tracking.TrackerKeys.KEY_ALLOW_GAP_CLOSING;
import static fiji.plugin.trackmate.tracking.TrackerKeys.KEY_ALLOW_TRACK_MERGING;
import static fiji.plugin.trackmate.tracking.TrackerKeys.KEY_ALLOW_TRACK_SPLITTING;
import static fiji.plugin.trackmate.tracking.TrackerKeys.KEY_ALTERNATIVE_LINKING_COST_FACTOR;
import static fiji.plugin.trackmate.tracking.TrackerKeys.KEY_BLOCKING_VALUE;
import static fiji.plugin.trackmate.tracking.TrackerKeys.KEY_CUTOFF_PERCENTILE;
import static fiji.plugin.trackmate.tracking.TrackerKeys.KEY_GAP_CLOSING_FEATURE_PENALTIES;
import static fiji.plugin.trackmate.tracking.TrackerKeys.KEY_GAP_CLOSING_MAX_DISTANCE;
import static fiji.plugin.trackmate.tracking.TrackerKeys.KEY_GAP_CLOSING_MAX_FRAME_GAP;
import static fiji.plugin.trackmate.tracking.TrackerKeys.KEY_LINKING_FEATURE_PENALTIES;
import static fiji.plugin.trackmate.tracking.TrackerKeys.KEY_LINKING_MAX_DISTANCE;
import static fiji.plugin.trackmate.tracking.TrackerKeys.KEY_MERGING_FEATURE_PENALTIES;
import static fiji.plugin.trackmate.tracking.TrackerKeys.KEY_MERGING_MAX_DISTANCE;
import static fiji.plugin.trackmate.tracking.TrackerKeys.KEY_SPLITTING_FEATURE_PENALTIES;
import static fiji.plugin.trackmate.tracking.TrackerKeys.KEY_SPLITTING_MAX_DISTANCE;
import static net.imglib2.type.numeric.ARGBType.alpha;
import static net.imglib2.type.numeric.ARGBType.blue;
import static net.imglib2.type.numeric.ARGBType.green;
import static net.imglib2.type.numeric.ARGBType.red;
import static net.imglib2.type.numeric.ARGBType.rgba;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.Shape;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Future;

import org.scijava.ItemIO;
import org.scijava.command.Command;
import org.scijava.command.CommandModule;
import org.scijava.log.LogService;
import org.scijava.module.Module;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import fiji.plugin.trackmate.Model;
import fiji.plugin.trackmate.Settings;
import fiji.plugin.trackmate.Spot;
import fiji.plugin.trackmate.SpotCollection;
import fiji.plugin.trackmate.TrackMate;
import fiji.plugin.trackmate.TrackModel;
import fiji.plugin.trackmate.detection.LogDetectorFactory;
import fiji.plugin.trackmate.features.FeatureFilter;
import fiji.plugin.trackmate.features.track.TrackDurationAnalyzer;
import fiji.plugin.trackmate.features.track.TrackSpeedStatisticsAnalyzer;
import fiji.plugin.trackmate.gui.TrackMateGUIController;
import fiji.plugin.trackmate.io.TmXmlWriter;
import fiji.plugin.trackmate.tracking.LAPUtils;
import fiji.plugin.trackmate.tracking.kalman.KalmanTrackerFactory;
import ij.IJ;
import ij.ImagePlus;
import ij.gui.Roi;
import ij.io.FileInfo;
import ij.io.FileOpener;
import ij.io.FileSaver;
import ij.io.Opener;
import ij.measure.ResultsTable;
import ij.plugin.ChannelSplitter;
import ij.plugin.filter.AVI_Writer;
import ij.plugin.filter.Analyzer;
import ij.plugin.frame.RoiManager;
import io.scif.SCIFIO;
import io.scif.services.DatasetIOService;
import net.imagej.Dataset;
import net.imagej.DatasetService;
import net.imagej.ImageJ;
import net.imagej.axis.CalibratedAxis;
import net.imagej.ops.OpService;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.ImagePlusAdapter;
import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.ARGBType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.view.Views;

@Plugin(type = Command.class, menuPath = "Plugins>Centrosomes Tracking")
public class CentrosomeTracking<T extends RealType<T>> implements Command {
	@Parameter
	private Dataset currentData;
	@Parameter
	private LogService logService;
	@Parameter
	private OpService opService;
	@Parameter
	private DatasetService datasetService;
	@Parameter
	private DatasetIOService datasetIOService;
	@Parameter
	private boolean showRender;

	// @Parameter
	private double contrastSaturation = 0.35f;
	private double thresholdMode = 1.0f;
	private double outlierRadius = 2.0f;
	private double outlierThreshold = 50.0f;
	private double trackmateLogRadius = 0.5f;
	private double trackmateLogThreshold = 20.0f;
	private double trackmateLogDoMedianFiltering = 1.0f;
	private double trackmateLogDoSubpixelLoc = 1.0f;
	private double trackmateLogSpotQuality = 5600.0f;
	private double trackmateTrackSpots = 6.0f;
	private double trackmateTrackDisplacement = 2.1f;
	private double trackmateLinkingMaxDist = 20.0f;
	private double trackmateLinkingAllowGapClosing = 0.0f;
	private double trackmateLinkingGapClosingMaxDist = 5.0f;
	private double trackmateLinkingGapClosingMaxFrameGap = 2.0f;

	protected String experimentTag = null;

	protected ImagePlus imagePlusDna;
	protected ImagePlus imagePlusCentrosomes;
	protected ImagePlus imagePlusMicrotubules;

	protected CalibratedAxis cax;
	protected CalibratedAxis cay;

	@Parameter(type = ItemIO.OUTPUT)
	private Dataset dataOut;

	protected TrackMate trkMtCentrosomes;
	protected Settings trkMtSettingsCentrosomes;
	protected Model trkMtModelCentrosomes;
	protected TrackMateGUIController guiTrackMateCentrosomes;

	protected TrackMate trkMtNuclei;
	protected Settings trkMtSettingsNuclei;
	protected Model trkMtModelNuclei;

	protected SpotCollection spotNucleusCentroid;
	protected Set<SpotCollection> spotCentrosomes;
	protected Set<SpotCollection> spotNuclei;

	protected String path;
	protected String filename;

	private double sigma;
	private double u1;
	private double u2;

	HashMap<Integer, MasterSheetItem> ms;

	@Override
	public void run() {
		Path p = Paths.get(currentData.getImgPlus().getSource());
		this.path = p.getParent().getParent().toString() + "/";
		this.filename = p.getName(p.getNameCount() - 1).toString();
		this.filename = filename.substring(0, filename.indexOf('.'));

		// create crops dir, and movies / data
		new File("../out/crops").mkdirs();
		new File(path + "data").mkdirs();
		new File(path + "movies").mkdirs();

		if (this.experimentTag == null)
			this.experimentTag = filename;

		this.ms = new HashMap<>();

		this.loadConfiguration();
		this.saveConfiguration();

		this.cax = currentData.axis(0);
		this.cay = currentData.axis(1);

		// open a file with ImageJ
		final ImagePlus imageInput = new Opener().openImage(p.toString());

		ImagePlus[] channels = ChannelSplitter.split(imageInput);
		this.imagePlusDna = channels[0];
		this.imagePlusMicrotubules = channels[1];
		this.imagePlusCentrosomes = channels[2];

		IJ.run(imagePlusMicrotubules, "Enhance Contrast", "saturated=" + contrastSaturation);

		this.spotCentrosomes = new HashSet<>();
		this.spotNuclei = new HashSet<>();
		this.spotNucleusCentroid = new SpotCollection();

		// -------------------
		// Starts processing
		// -------------------
		this.sigma = currentData.axis(0).rawValue(2.0); // 2um
		this.u1 = Math.sqrt(sigma) * 0.42;
		this.u2 = Math.sqrt(sigma) * 0.53;

		this.preProcessDNAImage();
		this.trackDNA();
		this.updateSheetWithNucleiTracks();

		// this.preProcessCentrosomeImage();
		this.trackCentrosomes();
		this.saveCentrosomeTrackMateXML();

		this.updateSheetWithCentrosomesTracks();
		this.generateCentrosomeStructure();

		this.renderResultTable();
		Img<ARGBType> renderCellAnot = this.renderCellAnotations(this.getCompositeCellImage());
		if (showRender)
			ImageJFunctions.show(renderCellAnot);

		try {
			AVI_Writer aw = new AVI_Writer();
			aw.writeImage(ImageJFunctions.wrap(renderCellAnot, "anotated"), path + "movies/" + filename + ".avi",
					AVI_Writer.JPEG_COMPRESSION, 70);
		} catch (IOException e) {
			e.printStackTrace();
		}

		// Img<ARGBType> imgBW = this.getBWCellImage();
		Img<ARGBType> imgBW = this.renderCellAnotations(this.getCompositeCellImage());
		TrackModel tmn = trkMtModelNuclei.getTrackModel();
		for (Integer nucId : tmn.trackIDs(true)) {
			// imgBW = this.renderCellMarkers(imgBW, nucId + 1);
			this.extractNucleusCrop(imgBW, nucId + 1);
		}
		if (showRender)
			ImageJFunctions.show(imgBW);
	}

	protected void saveConfiguration() {
		File f = new File(path + "centrosomes_experiments.csv");
		ResultsTable rt;
		if (f.exists())
			rt = ResultsTable.open2(path + "centrosomes_experiments.csv");
		else
			rt = new ResultsTable();

		for (int row = 0; row < rt.size(); row++) {
			String itTag = rt.getStringValue("TAG", row);
			if (itTag.equals(experimentTag)) {
				rt.deleteRow(row);
			}
		}
		rt.incrementCounter();
		rt.addValue("TAG", experimentTag);
		rt.addValue("TRACKING_CONTRAST_SATURATION", contrastSaturation);
		rt.addValue("TRACKING_THRESHOLD_METHOD", thresholdMode);
		rt.addValue("DETECTION_OUTLIER_RADIUS", outlierRadius);
		rt.addValue("DETECTION_OUTLIER_THRESHOLD", outlierThreshold);
		rt.addValue("TRACKMATE_LOG_DETECTOR_RADIUS", trackmateLogRadius);
		rt.addValue("TRACKMATE_LOG_DETECTOR_THRESHOLD", trackmateLogThreshold);
		rt.addValue("TRACKMATE_LOG_DETECTOR_DO_MEDIAN_FILTERING", trackmateLogDoMedianFiltering);
		rt.addValue("TRACKMATE_LOG_DETECTOR_DO_SUBPIXEL_LOCALIZATION", trackmateLogDoSubpixelLoc);
		rt.addValue("TRACKMATE_LOG_FILTER_SPOT_QUALITY", trackmateLogSpotQuality);
		rt.addValue("TRACKMATE_TRACK_DISP", trackmateTrackDisplacement);
		rt.addValue("TRACKMATE_TRACK_SPOTS", trackmateTrackSpots);
		rt.addValue("TRACKMATE_SIMPLE_SPARSE_LAP_TRACKER_LINKING_MAX_DISTANCE", trackmateLinkingMaxDist);
		rt.addValue("TRACKMATE_SIMPLE_SPARSE_LAP_TRACKER_ALLOW_GAP_CLOSING", trackmateLinkingAllowGapClosing);
		rt.addValue("TRACKMATE_GAP_MAX_DISTANCE", trackmateLinkingGapClosingMaxDist);
		rt.addValue("TRACKMATE_SIMPLE_SPARSE_LAP_TRACKER_MAX_FRAME_GAP", trackmateLinkingGapClosingMaxFrameGap);
		rt.save(path + "centrosomes_experiments.csv");
	}

	protected boolean loadConfiguration() {
		File f = new File(path + "centrosomes_experiments.csv");
		if (!f.exists())
			return false;

		ResultsTable rt = ResultsTable.open2(path + "centrosomes_experiments.csv");
		for (int row = 0; row < rt.size(); row++) {
			String itTag = rt.getStringValue("TAG", row);
			if (itTag.equals(experimentTag)) {
				try {
					contrastSaturation = rt.getValue("TRACKING_CONTRAST_SATURATION", row);
					thresholdMode = rt.getValue("TRACKING_THRESHOLD_METHOD", row);
					outlierRadius = rt.getValue("DETECTION_OUTLIER_RADIUS", row);
					outlierThreshold = rt.getValue("DETECTION_OUTLIER_THRESHOLD", row);
					trackmateLogRadius = rt.getValue("TRACKMATE_LOG_DETECTOR_RADIUS", row);
					trackmateLogThreshold = rt.getValue("TRACKMATE_LOG_DETECTOR_THRESHOLD", row);
					trackmateLogDoMedianFiltering = rt.getValue("TRACKMATE_LOG_DETECTOR_DO_MEDIAN_FILTERING", row);
					trackmateLogDoSubpixelLoc = rt.getValue("TRACKMATE_LOG_DETECTOR_DO_SUBPIXEL_LOCALIZATION", row);
					trackmateLogSpotQuality = rt.getValue("TRACKMATE_LOG_FILTER_SPOT_QUALITY", row);
					trackmateTrackDisplacement = rt.getValue("TRACKMATE_TRACK_DISP", row);
					trackmateTrackSpots = rt.getValue("TRACKMATE_TRACK_SPOTS", row);
					trackmateLinkingMaxDist = rt.getValue("TRACKMATE_SIMPLE_SPARSE_LAP_TRACKER_LINKING_MAX_DISTANCE",
							row);
					trackmateLinkingAllowGapClosing = rt
							.getValue("TRACKMATE_SIMPLE_SPARSE_LAP_TRACKER_ALLOW_GAP_CLOSING", row);
					trackmateLinkingGapClosingMaxDist = rt.getValue("TRACKMATE_GAP_MAX_DISTANCE", row);
					trackmateLinkingGapClosingMaxFrameGap = rt
							.getValue("TRACKMATE_SIMPLE_SPARSE_LAP_TRACKER_MAX_FRAME_GAP", row);
				} catch (IllegalArgumentException iae) {
					logService.log().warn(iae.getMessage());
					logService.log().warn("STOPPING LOAD OF PARAMETERS");
				}
				return true;
			}
		}
		return false;
	}

	protected void preProcessDNAImage() {
		IJ.run(imagePlusDna, "Enhance Contrast", "saturated=" + contrastSaturation);
		switch (new Double(thresholdMode).intValue()) {
		case 1:
			IJ.run(imagePlusDna, "Convert to Mask", "method=Li background=Dark calculate black");
			break;
		case 2:
			IJ.run(imagePlusDna, "Convert to Mask", "method=Otsu background=Dark calculate black");
			break;
		}
		// IJ.run(imagePlusDna, "Close-", "stack");
		IJ.run(imagePlusDna, "Fill Holes", "stack");
		IJ.run(imagePlusDna, "Remove Outliers...",
				"radius=" + outlierRadius + " threshold=" + outlierThreshold + " which=Bright stack");
		IJ.run(imagePlusDna, "Distance Map", "stack");
		IJ.run(imagePlusDna, "Enhance Contrast", "saturated=" + contrastSaturation);
		IJ.run(imagePlusDna, "Gaussian Blur...", "sigma=2.0 scaled stack");
		IJ.run(imagePlusDna, "Enhance Contrast", "saturated=" + contrastSaturation);
		IJ.run(imagePlusDna, "Convert to Mask", "method=Otsu background=Dark calculate black");
		IJ.run(imagePlusDna, "Watershed", "stack");
	}

	protected void preProcessCentrosomeImage() {
		IJ.run(imagePlusCentrosomes, "Enhance Contrast", "saturated=" + contrastSaturation);
		IJ.run(imagePlusCentrosomes, "Convert to Mask", "method=Minimum background=Dark calculate black");
		IJ.run(imagePlusCentrosomes, "Remove Outliers...", "radius=1 threshold=10 which=Bright stack");
	}

	private void runTrackMateOnDNA() {
		logService.info("Starting nuclei tracking.");
		imagePlusDna.setOpenAsHyperStack(true);
		// configure TrackMate
		trkMtSettingsNuclei = new Settings();
		trkMtSettingsNuclei.setFrom(imagePlusDna);
		trkMtModelNuclei = new Model();
		final String spaceUnits = trkMtSettingsNuclei.imp.getCalibration().getXUnit();
		final String timeUnits = trkMtSettingsNuclei.imp.getCalibration().getTimeUnit();
		trkMtModelNuclei.setPhysicalUnits(spaceUnits, timeUnits);

		// Configure detector
		trkMtSettingsNuclei.detectorFactory = new LogDetectorFactory<DoubleType>();
		final Map<String, Object> detctorMap = new HashMap<String, Object>();
		detctorMap.put("DO_SUBPIXEL_LOCALIZATION", true);
		detctorMap.put("RADIUS", 20.0);
		detctorMap.put("TARGET_CHANNEL", 0);
		detctorMap.put("THRESHOLD", 0.0);
		detctorMap.put("DO_MEDIAN_FILTERING", false);
		trkMtSettingsNuclei.detectorSettings = detctorMap;

		// Configure tracker
		trkMtSettingsNuclei.trackerFactory = new KalmanTrackerFactory();
		trkMtSettingsNuclei.trackerSettings = LAPUtils.getDefaultLAPSettingsMap();
		trkMtSettingsNuclei.trackerSettings.put("KALMAN_SEARCH_RADIUS", 15.0);
		trkMtSettingsNuclei.trackerSettings.replace("ALLOW_TRACK_SPLITTING", false);
		trkMtSettingsNuclei.trackerSettings.replace("ALLOW_TRACK_MERGING", false);

		// -------------------
		// Instantiate plugin
		// -------------------
		trkMtNuclei = new TrackMate(trkMtModelNuclei, trkMtSettingsNuclei);

		// --------
		// Process
		// --------
		if (!trkMtNuclei.checkInput())
			IJ.log("Error on input for nuclei TrackMate!");

		// ----------------------------------------
		// Add nuclei centrosomes for tracking
		// ----------------------------------------
		for (Iterator<Spot> nspot = spotNucleusCentroid.iterator(true); nspot.hasNext();) {
			Spot n = nspot.next();
			int frame = n.getFeature(Spot.FRAME).intValue();
			trkMtModelNuclei.addSpotTo(n, frame);
		}

		boolean processSuccessful = true;
		processSuccessful = processSuccessful && trkMtNuclei.execTracking();
		processSuccessful = processSuccessful && trkMtNuclei.computeTrackFeatures(true);
		processSuccessful = processSuccessful && trkMtNuclei.execTrackFiltering(true);
		processSuccessful = processSuccessful && trkMtNuclei.computeEdgeFeatures(true);

		if (!processSuccessful)
			logService.error("Error on TrackMate nuclei processing!");
		else {
			logService.info("Nuclei tracking processed!");
		}
	}

	protected void trackDNA() {
		logService.info("Starting nuclei recognition.");
		IJ.run(imagePlusDna, "Analyze Particles...", "size=50-Infinity show=Overlay clear record add stack");

		RoiManager manager = RoiManager.getInstance();
		// sorts nucleus ROIs based on center of mass position
		if (manager != null) {
			Roi[] rois = manager.getRoisAsArray();
			int k = 0;
			for (Roi roi : rois) {
				Rectangle bbox = roi.getBounds();
				int frame = roi.getPosition() - 1;
				String new_name = String.format("%04d-%04d-%04d", frame, bbox.y, bbox.x);
				roi.setName(new_name);
				manager.rename(k, new_name);
				k++;
			}
			manager.runCommand("sort");

			// rename rois based on frame-correlative
			k = 0;
			int j = 1;
			int frame = 0;
			rois = manager.getRoisAsArray();
			for (Roi roi : rois) {
				double[] centroid = roi.getContourCentroid();
				if (frame < (roi.getPosition() - 1))
					j = 1;
				frame = roi.getPosition() - 1;
				String new_name = String.format("%04d-%04d", frame, j);
				logService.debug(
						"Renaming nuclei roi at frame:" + frame + " name:" + roi.getName() + " new name:" + new_name);
				roi.setName(new_name);
				manager.rename(k, new_name);

				Spot spotCentroid = new Spot(imagePlusDna.getCalibration().getX(centroid[0]),
						imagePlusDna.getCalibration().getY(centroid[1]), 0.0, 1.0, 0.0, new_name);
				// adds tag id in two features for later when the master sheet
				// is builded
				spotCentroid.putFeature("NucleiIDframe", (double) frame);
				spotCentroid.putFeature("NucleiIDcorr", (double) j);

				spotNucleusCentroid.add(spotCentroid, frame);
				k++;
				j++;
			}
		}
		this.runTrackMateOnDNA();
	}

	protected void updateSheetWithNucleiTracks() {
		logService.info("Updating sheet with nuclei tracks");
		TrackModel tmn = trkMtModelNuclei.getTrackModel();

		// add nuclei 0 (no nuclei) by definition
		MasterSheetItem mi = new MasterSheetItem();
		mi.nuclei_roi_id = "no-nuclei";
		mi.first_frame = 0;
		mi.centrosome_spot_list = new ArrayList<Spot>();
		// mi.rois_in_the_same_track = new ArrayList<String>();
		// mi.rois_in_the_same_track.add(String.format("%04d-%04d", 0, 0));
		ms.put(0, mi);

		// Idea is to get first point for each track and search for the nearest
		// nuclei centroid. After that, rename all points in the track with the
		// same id, different frame.
		for (Integer trkID : tmn.trackIDs(true)) {
			List<Spot> trackSpotsList = new ArrayList<Spot>(tmn.trackSpots(trkID));
			java.util.Collections.sort(trackSpotsList, Spot.frameComparator);

			Spot firstSpot = trackSpotsList.get(0);
			int frame = firstSpot.getFeature(Spot.FRAME).intValue();
			logService.debug(trkID + ": " + tmn.name(trkID) + " frame: " + frame);

			// search for closest nuclei from firstSpot in frame
			RoiManager manager = RoiManager.getInstance();
			// get closest ROI name
			double min_dist = Double.MAX_VALUE;
			String min_roi_id = "";
			if (manager != null) {
				Roi[] rois = manager.getRoisAsArray();
				for (Roi roi : rois) {
					int frameroi = roi.getPosition() - 1;
					if (frame == frameroi) {
						double[] centroid = roi.getContourCentroid();
						Spot spotCentroid = new Spot(imagePlusDna.getCalibration().getX(centroid[0]),
								imagePlusDna.getCalibration().getY(centroid[1]), 0.0, 1.0, 0.0, roi.getName());
						if (firstSpot.squareDistanceTo(spotCentroid) < min_dist) {
							min_dist = firstSpot.squareDistanceTo(spotCentroid);
							min_roi_id = roi.getName();
						}
					}
				}
			}

			MasterSheetItem i = new MasterSheetItem();
			i.nuclei_roi_id = min_roi_id;
			i.first_frame = frame;
			i.centrosome_spot_list = new ArrayList<Spot>();

			// adds the list or rois from the track based in the metadata
			// that we added in trackDNA()
			i.rois_in_the_same_track = new ArrayList<String>();
			for (Spot sr : trackSpotsList) {
				i.rois_in_the_same_track.add(String.format("%04d-%04d", sr.getFeature("NucleiIDframe").intValue(),
						sr.getFeature("NucleiIDcorr").intValue()));
			}
			ms.put(trkID, i);
		}
	}

	protected void trackCentrosomes() {
		logService.info("Starting centrosome recognition and tracking.");
		imagePlusCentrosomes.setOpenAsHyperStack(true);
		// configure TrackMate
		trkMtSettingsCentrosomes = new Settings();
		trkMtSettingsCentrosomes.setFrom(imagePlusCentrosomes);
		trkMtModelCentrosomes = new Model();
		final String spaceUnits = trkMtSettingsCentrosomes.imp.getCalibration().getXUnit();
		final String timeUnits = trkMtSettingsCentrosomes.imp.getCalibration().getTimeUnit();
		trkMtModelCentrosomes.setPhysicalUnits(spaceUnits, timeUnits);

		// Configure detector
		trkMtSettingsCentrosomes.detectorFactory = new LogDetectorFactory<DoubleType>();
		final Map<String, Object> detctorMap = new HashMap<String, Object>();
		detctorMap.put("RADIUS", trackmateLogRadius);
		detctorMap.put("TARGET_CHANNEL", 0);
		detctorMap.put("THRESHOLD", trackmateLogThreshold);
		detctorMap.put("DO_MEDIAN_FILTERING", trackmateLogDoMedianFiltering == 1 ? true : false);
		detctorMap.put("DO_SUBPIXEL_LOCALIZATION", trackmateLogDoSubpixelLoc == 1 ? true : false);
		trkMtSettingsCentrosomes.detectorSettings = detctorMap;

		trkMtSettingsCentrosomes.addSpotFilter(new FeatureFilter("QUALITY", trackmateLogSpotQuality, true));

		// Configure tracker - We don't want to allow merges and fusions
		trkMtSettingsCentrosomes.trackerFactory = new KalmanTrackerFactory();
		trkMtSettingsCentrosomes.trackerSettings = LAPUtils.getDefaultLAPSettingsMap();
		trkMtSettingsCentrosomes.trackerSettings.put("KALMAN_SEARCH_RADIUS", trackmateLinkingGapClosingMaxDist);

		// Linking
		// trkMtSettingsCentrosomes.trackerSettings.replace(TrackerKeys.KEY_LINKING_MAX_DISTANCE,
		// );
		// trkMtSettingsCentrosomes.trackerSettings.replace(TrackerKeys.KEY_LINKING_FEATURE_PENALTIES,
		// );

		// Gap closing
		trkMtSettingsCentrosomes.trackerSettings.replace(KEY_ALLOW_GAP_CLOSING,
				trackmateLinkingAllowGapClosing == 1 ? true : false);
		trkMtSettingsCentrosomes.trackerSettings.replace(KEY_GAP_CLOSING_MAX_DISTANCE,
				trackmateLinkingGapClosingMaxDist);
		trkMtSettingsCentrosomes.trackerSettings.replace(KEY_GAP_CLOSING_MAX_FRAME_GAP,
				(int) trackmateLinkingGapClosingMaxFrameGap);
		// trkMtSettingsCentrosomes.trackerSettings.replace(TrackerKeys.KEY_GAP_CLOSING_FEATURE_PENALTIES,
		// );

		// Track splitting
		trkMtSettingsCentrosomes.trackerSettings.replace(KEY_ALLOW_TRACK_SPLITTING, false);
		// trkMtSettingsCentrosomes.trackerSettings.replace(TrackerKeys.KEY_SPLITTING_MAX_DISTANCE,
		// );
		// trkMtSettingsCentrosomes.trackerSettings.replace(TrackerKeys.KEY_SPLITTING_FEATURE_PENALTIES,
		// );

		// Track merging
		trkMtSettingsCentrosomes.trackerSettings.replace(KEY_ALLOW_TRACK_MERGING, false);
		// trkMtSettingsCentrosomes.trackerSettings.replace(TrackerKeys.KEY_MERGING_MAX_DISTANCE,
		// );
		// trkMtSettingsCentrosomes.trackerSettings.replace(TrackerKeys.KEY_MERGING_FEATURE_PENALTIES,
		// );

		// Others
		// trkMtSettingsCentrosomes.trackerSettings.replace(TrackerKeys.KEY_BLOCKING_VALUE,
		// );
		// trkMtSettingsCentrosomes.trackerSettings.replace(TrackerKeys.KEY_ALTERNATIVE_LINKING_COST_FACTOR,
		// );
		// trkMtSettingsCentrosomes.trackerSettings.replace(TrackerKeys.KEY_CUTOFF_PERCENTILE,
		// );

		trkMtSettingsCentrosomes
				.addTrackFilter(new FeatureFilter("TRACK_DISPLACEMENT", trackmateTrackDisplacement, true));
		trkMtSettingsCentrosomes.addTrackFilter(new FeatureFilter("NUMBER_SPOTS", trackmateTrackSpots, true));

		trkMtSettingsCentrosomes.addTrackAnalyzer(new TrackDurationAnalyzer());
		trkMtSettingsCentrosomes.addTrackAnalyzer(new TrackSpeedStatisticsAnalyzer());

		// -------------------
		// Instantiate plugin
		// -------------------
		trkMtCentrosomes = new TrackMate(trkMtModelCentrosomes, trkMtSettingsCentrosomes);

		// --------
		// Process
		// --------
		if (!trkMtCentrosomes.checkInput())
			logService.error("Error on input for centrosomes TrackMate! " + trkMtCentrosomes.getErrorMessage());
		if (!trkMtCentrosomes.process())
			logService.error("Error on TrackMate centrosomes processing! " + trkMtCentrosomes.getErrorMessage());
		else {
			logService.info("Centrosomes tracking processed!");
		}
	}

	protected void updateSheetWithCentrosomesTracks() {
		logService.info("Renaming centrosomes tracks");

		TrackModel tmc = trkMtModelCentrosomes.getTrackModel();
		// For each centrosome track, search for it's nearest nuclei at the end
		for (Integer cenTrkId : tmc.trackIDs(true)) {
			logService.debug("looking centrosome track id " + cenTrkId);
			List<Spot> trackSpotsList = new ArrayList<Spot>(tmc.trackSpots(cenTrkId));
			java.util.Collections.sort(trackSpotsList, Spot.frameComparator);

			// criteria: search for closest nuclei in first and last frame. If
			// they are equal, assign it, otherwise, do nothing.
			SpotCollection nucCol = this.getNucleiCentroids();
			Spot firstSpot = trackSpotsList.get(0);
			Spot lastSpot = trackSpotsList.get(trackSpotsList.size() - 1);
			int firstFrame = firstSpot.getFeature(Spot.FRAME).intValue();
			int lastFrame = lastSpot.getFeature(Spot.FRAME).intValue();
			// Nuclei is final_id
			int firstNucTrkID = nucCol.getClosestSpot(firstSpot, firstFrame, true).getFeature("Nuclei").intValue();
			int lastNucTrkID = nucCol.getClosestSpot(lastSpot, lastFrame, true).getFeature("Nuclei").intValue();

			Set<Integer> nuc_set = new HashSet<Integer>();
			for (Spot s : trackSpotsList) {
				Spot closeS = nucCol.getClosestSpot(s, s.getFeature(Spot.FRAME).intValue(), true);

				// get rid of spots too far away
				if (s.squareDistanceTo(closeS) < 200) // 10 sqrt(2)
					nuc_set.add(closeS.getFeature("Nuclei").intValue());
			}

			// Criteria: 1- If first and last nuclei are the same, assign
			// centrosome to it.
			int nucFinalId;
			if (firstNucTrkID == lastNucTrkID)
				nucFinalId = firstNucTrkID;
			// 2- if the nuclei is the same along the centrosome track,
			// assign it
			else if (nuc_set.size() == 1)
				nucFinalId = nuc_set.iterator().next();
			else
				nucFinalId = 0;

			// search master sheet for nuclei id and add centrosome
			MasterSheetItem msi = MasterSheetItem.getMasterSheetItemFromFinalId(ms.values().iterator(), nucFinalId);
			if (msi != null) {
				// working on the right track: add centrosome list
				if (nucFinalId > 0) {
					for (Spot s : trackSpotsList) {
						s.putFeature("Centrosome", (double) (msi.final_id * 100 + msi.n_centrosomes));
						s.putFeature("Nuclei", (double) msi.final_id);
					}
				} else {
					for (Spot s : trackSpotsList) {
						s.putFeature("Centrosome", (double) msi.n_centrosomes);
						s.putFeature("Nuclei", 0.0);
					}
				}

				msi.n_centrosomes++;
				msi.centrosome_spot_list.addAll(trackSpotsList);
			}
		}
	}

	public void saveCentrosomeTrackMateXML() {
		// Save results into XML
		final File file = new File(path + filename + "-TrackMate.xml");

		// Write model, settings and GUI state
		final TmXmlWriter writer = new TmXmlWriter(file);

		writer.appendModel(trkMtCentrosomes.getModel());
		writer.appendSettings(trkMtCentrosomes.getSettings());

		try {
			writer.writeToFile();
			logService.info("Data saved to: " + file.toString() + '\n');
		} catch (final FileNotFoundException e) {
			logService.error("File not found:\n" + e.getMessage() + '\n');
			return;
		} catch (final IOException e) {
			logService.error("Input/Output error:\n" + e.getMessage() + '\n');
			return;
		}
	}

	private SpotCollection getNucleiCentroids() {
		// first step: sort and rename rois and tracks.
		// get all tracks starting on a frame
		int trk_corr = 1;
		for (int f = 0; f < currentData.getFrames(); f++) {
			HashMap<Integer, MasterSheetItem> frame_msi = new HashMap<>();
			for (Integer msi_id : ms.keySet()) {
				MasterSheetItem msi = ms.get(msi_id);
				if (msi.first_frame == f) {
					frame_msi.put(msi_id, msi);
				}
			}

			// create and sort list based on roi id
			List<MasterSheetItem> msiByRoiId = new ArrayList<MasterSheetItem>(frame_msi.values());

			Collections.sort(msiByRoiId, new Comparator<MasterSheetItem>() {
				public int compare(MasterSheetItem o1, MasterSheetItem o2) {
					int roi_id1 = Integer.valueOf(o1.nuclei_roi_id.substring(5, 9));
					int roi_id2 = Integer.valueOf(o2.nuclei_roi_id.substring(5, 9));
					return roi_id1 - roi_id2;
				}
			});

			// iterate over sorted list and assign final id to each track
			for (MasterSheetItem msi : msiByRoiId) {
				msi.final_id = trk_corr;
				trk_corr++;
			}
		}

		// build nucleus centroid spot collection
		TrackModel tmn = trkMtModelNuclei.getTrackModel();
		SpotCollection nucleiCollectOut = new SpotCollection();
		for (Integer trkID : tmn.trackIDs(true)) {
			// search master sheet for nuclei id and add centrosome
			if (ms.containsKey(trkID)) {
				MasterSheetItem msi = ms.get(trkID);
				// working on the right track: extract every point of the
				// track list
				double nuclei_id = msi.final_id;
				for (Iterator<Spot> nucSpot = tmn.trackSpots(trkID).iterator(); nucSpot.hasNext();) {
					Spot nuSpot = nucSpot.next();
					nuSpot.putFeature("Nuclei", nuclei_id);
					nucleiCollectOut.add(nuSpot, nuSpot.getFeature(Spot.FRAME).intValue());
				}
			}
		}
		return nucleiCollectOut;
	}

	private void generateCentrosomeStructure() {
		// -------------------
		// Create list
		// -------------------
		for (MasterSheetItem msi : ms.values()) {
			SpotCollection cSC = new SpotCollection();
			for (Spot s : msi.centrosome_spot_list) {
				int frame = s.getFeature(Spot.FRAME).intValue();
				if (msi.final_id == 0) {
					// adds the non-recognized centrosome tracks
					s.putFeature("NuclX", -1.0);
					s.putFeature("NuclY", -1.0);
					s.putFeature("WhereInNuclei", Double.MIN_VALUE);
					s.putFeature("ValidCentroid", -1.0);
				} else {
					Roi roi = CentrTrackRoiUtils.getRoiForFrame(msi.rois_in_the_same_track, frame);
					if (roi == null)
						continue;
					double[] centroid = roi.getContourCentroid();

					s.putFeature("NuclX", cax.calibratedValue(centroid[0]));
					s.putFeature("NuclY", cay.calibratedValue(centroid[1]));

					AffineTransform scaleMatrix = new AffineTransform();
					scaleMatrix.translate(centroid[0], centroid[1]);
					scaleMatrix.scale(u1, u1);
					scaleMatrix.translate(-centroid[0], -centroid[1]);
					boolean isInside1 = scaleMatrix.createTransformedShape(roi.getPolygon()).contains(
							cax.rawValue(s.getFeature(Spot.POSITION_X)), cax.rawValue(s.getFeature(Spot.POSITION_Y)));
					scaleMatrix = new AffineTransform();
					scaleMatrix.translate(centroid[0], centroid[1]);
					scaleMatrix.scale(u2, u2);
					scaleMatrix.translate(-centroid[0], -centroid[1]);
					boolean isInside2 = scaleMatrix.createTransformedShape(roi.getPolygon()).contains(
							cax.rawValue(s.getFeature(Spot.POSITION_X)), cax.rawValue(s.getFeature(Spot.POSITION_Y)));

					// 0:out, 1:touching, 2:inside
					double whereInNuclei = Double.MIN_VALUE;
					if (!isInside1 && !isInside2)
						whereInNuclei = 0;
					if (!isInside1 && isInside2)
						whereInNuclei = 1;
					if (isInside1 && isInside2)
						whereInNuclei = 2;

					s.putFeature("WhereInNuclei", whereInNuclei);
					s.putFeature("ValidCentroid", CentrTrackRoiUtils.isRoiTouchingEdge(roi, imagePlusDna) ? 0.0 : 1.0);
				}
				cSC.add(s, frame);
			}
			spotCentrosomes.add(cSC);
		}
	}

	@SuppressWarnings("unchecked")
	public Img<ARGBType> getCompositeCellImage() {
		logService.info("Getting composite image of all channels");
		// --------------------------------
		// generate a RGB sandwich of all channels
		// --------------------------------
		long frames = currentData.getFrames();
		long width = currentData.getWidth();
		long height = currentData.getHeight();
		Img<UnsignedShortType> data = (Img<UnsignedShortType>) currentData.getImgPlus().getImg();

		// create the ImgFactory
		final ImgFactory<ARGBType> imgFactory = new ArrayImgFactory<ARGBType>();
		// create an Img with dimensions equals to
		int[] dims = { (int) width, (int) height, 1, (int) frames };
		final Img<ARGBType> compImg = imgFactory.create(dims, new ARGBType());

		// create cursors
		final RandomAccess<ARGBType> cComp = compImg.randomAccess();
		final RandomAccess<UnsignedShortType> cd = data.randomAccess();
		// final long[] p = new long[currentData.numDimensions()];
		double maxShort = cd.get().getMaxValue();
		double red_a = 255.0d / maxShort * 2.0d;
		double green_a = 255.0d / maxShort * 1.0d;
		double blue_a = 255.0d / maxShort * 3.0d;
		// iterate over the input cursor
		for (int f = 0; f < frames; f++) {
			for (int w = 0; w < width; w++)
				for (int h = 0; h < height; h++) {
					// move input cursor
					cd.setPosition(new int[] { w, h, 0, f });
					UnsignedShortType u1 = cd.get().copy(); // dna

					cd.setPosition(new int[] { w, h, 1, f });
					UnsignedShortType u2 = cd.get().copy(); // mt

					cd.setPosition(new int[] { w, h, 2, f });
					UnsignedShortType u3 = cd.get().copy(); // centrosomes

					// set the output cursor to the position of the input cursor
					cComp.setPosition(new int[] { w, h, 0, f });

					// set the value of this pixel of the output image, every
					// Type supports T.set( T type )
					cComp.get().set(rgba(Math.min(255, u1.get() * red_a), Math.min(255, u2.get() * green_a),
							Math.min(255, u3.get() * blue_a), 255));
				}
		}
		return compImg;
	}

	@SuppressWarnings("unchecked")
	public Img<ARGBType> getBWCellImage() {
		logService.info("Getting black & white image of all channels");
		// --------------------------------
		// generate a RGB sandwich of all channels
		// --------------------------------
		long frames = currentData.getFrames();
		long width = currentData.getWidth();
		long height = currentData.getHeight();
		Img<UnsignedShortType> data = (Img<UnsignedShortType>) currentData.getImgPlus().getImg();

		// create the ImgFactory
		final ImgFactory<ARGBType> imgFactory = new ArrayImgFactory<ARGBType>();
		// create an Img with dimensions equals to
		int[] dims = { (int) width, (int) height, 1, (int) frames };
		final Img<ARGBType> compImg = imgFactory.create(dims, new ARGBType());

		// create cursors
		final RandomAccess<ARGBType> cComp = compImg.randomAccess();
		final RandomAccess<UnsignedShortType> cd = data.randomAccess();
		// final long[] p = new long[currentData.numDimensions()];
		double maxShort = cd.get().getMaxValue();
		double bw_a = 255.0d / maxShort * 1.0d;
		double bw_b = 255.0d / maxShort * 7.0d;
		// iterate over the input cursor
		for (int f = 0; f < frames; f++) {
			for (int w = 0; w < width; w++)
				for (int h = 0; h < height; h++) {
					// move input cursor
					cd.setPosition(new int[] { w, h, 0, f });
					UnsignedShortType u1 = cd.get().copy(); // dn

					cd.setPosition(new int[] { w, h, 1, f });
					UnsignedShortType u2 = cd.get().copy(); // mt

					// set the output cursor to the position of the input cursor
					cComp.setPosition(new int[] { w, h, 0, f });

					cd.setPosition(new int[] { w, h, 2, f });
					UnsignedShortType u3 = cd.get().copy(); // centrosomes

					// set the value of this pixel of the output image, every
					// Type supports T.set( T type )
					cComp.get().set(rgba(Math.min(255, u1.get() * bw_a), 0, Math.min(255, u3.get() * bw_b), 255));
				}
		}
		return compImg;
	}

	public void extractNucleusCrop(Img<ARGBType> img, int nucId) {
		logService.info("Extracting crops");
		if (!ms.containsKey(nucId)) {
			logService.error("No rois for that nuclei track id");
			return;
		}

		long width = currentData.getWidth();
		long height = currentData.getHeight();
		long frames = currentData.getFrames();

		// --------------------------------
		// extract nuclei's ROI
		// --------------------------------
		int maxWidth = 0;
		int maxHeight = 0;
		double[] minCxy = { Double.MAX_VALUE, Double.MAX_VALUE };
		double[] maxCxy = { Double.MIN_VALUE, Double.MIN_VALUE };

		RoiManager manager = RoiManager.getInstance();
		Roi[] roi_list = manager.getRoisAsArray();

		MasterSheetItem msi = ms.get(nucId);
		for (String roi_id : msi.rois_in_the_same_track) {
			for (Roi roi : roi_list) {
				if (roi.getName().equals(roi_id)) {
					Rectangle r = roi.getBounds();
					maxWidth = Math.max(maxWidth, new Double(r.getWidth()).intValue());
					maxHeight = Math.max(maxHeight, new Double(r.getHeight()).intValue());
					double[] centroid = roi.getContourCentroid();
					minCxy[0] = Math.min(minCxy[0], centroid[0]);
					minCxy[1] = Math.min(minCxy[1], centroid[1]);
					maxCxy[0] = Math.max(minCxy[0], centroid[0]);
					maxCxy[1] = Math.max(minCxy[1], centroid[1]);
				}
			}
		}
		// maxWidth = Math.max(maxWidth, maxHeight);
		// maxHeight = Math.max(maxWidth, maxHeight);
		// maxWidth += 40;
		// maxHeight += 40;
		maxWidth = 200;
		maxHeight = 200;

		if (minCxy[0] - maxWidth / 2 <= 0 || minCxy[1] - maxHeight / 2 <= 0 || maxCxy[0] + maxWidth / 2 >= width
				|| maxCxy[1] + maxHeight / 2 >= height)
			return;

		// --------------------------------
		// Draw ROIs in final results image
		// --------------------------------
		// create result image
		final ImgFactory<ARGBType> imgFactory = new ArrayImgFactory<ARGBType>();
		// create an Img with dimensions equals to
		int[] dims = { maxWidth, maxHeight, 1, (int) frames };
		final Img<ARGBType> cropImg = imgFactory.create(dims, new ARGBType());

		for (String roi_id : msi.rois_in_the_same_track) {
			for (Roi roi : roi_list) {
				if (roi.getName().equals(roi_id)) {
					int frame = Integer.valueOf(roi_id.substring(0, 4));
					double[] centroid = roi.getContourCentroid();

					int cx = (int) centroid[0] - maxWidth / 2;
					int cy = (int) centroid[1] - maxHeight / 2;

					long[] min = { 0, 0, 0, frame };
					long[] max = { width - 1, height - 1, 0, frame };

					logService.debug("Extract: " + Arrays.toString(min) + " " + Arrays.toString(max)
							+ String.format(" %d %d", cx, cy));
					// use a View to define an interval (min and max
					// coordinate,inclusive) to display
					RandomAccessibleInterval<ARGBType> view = Views.interval(img, min, max);

					// copy data into cropImg
					final RandomAccess<ARGBType> cCrop = cropImg.randomAccess();
					final RandomAccess<ARGBType> cView = view.randomAccess();
					for (int w = 0; w < maxWidth; w++)
						for (int h = 0; h < maxHeight; h++) {
							// move input cursor
							cCrop.setPosition(new int[] { w, h, 0, frame });
							cView.setPosition(new int[] { cx + w, cy + h, 0, frame });
							cCrop.get().set(cView.get());
						}
				}
			}
		}

		// save
		String fname = "../data/crops/" + filename + "-crop-C" + nucId + ".tif";
		// new FileSaver(ImageJFunctions.wrap(cropImg, "")).saveAsTiff(fname);
		// ImageJFunctions.show(cropImg);
		try {
			AVI_Writer aw = new AVI_Writer();
			aw.writeImage(ImageJFunctions.wrap(cropImg, "cropped"), fname + ".avi", AVI_Writer.PNG_COMPRESSION, 100);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private Img<ARGBType> renderCellAnotations(Img<ARGBType> imgOut) {
		int width = (int) imgOut.dimension(0);
		int height = (int) imgOut.dimension(1);
		long frames = imgOut.dimension(3);
		long[] min = new long[imgOut.numDimensions()];
		long[] max = new long[imgOut.numDimensions()];
		imgOut.min(min);
		imgOut.max(max);
		logService.debug(Arrays.toString(min) + " " + Arrays.toString(max));

		for (int f = 0; f < frames; f++) {
			long[] mini = new long[] { min[0], min[1], min[2], f };
			long[] maxi = new long[] { max[0], max[1], max[2], f };

			RandomAccessibleInterval<ARGBType> vwOut = Views.interval(imgOut, mini, maxi);
			ImagePlus imp = ImageJFunctions.wrap(vwOut, "");

			BufferedImage bi = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
			Graphics2D g = (Graphics2D) bi.getGraphics();
			g.drawImage(imp.getImage(), 0, 0, null);

			// plots all information that can be plotted from master sheet:
			// rois, centrosomes, nuclei id's.
			for (MasterSheetItem msi : ms.values()) {
				// ROIs
				Roi roi = CentrTrackRoiUtils.getRoiForFrame(msi.rois_in_the_same_track, f);
				if (roi != null) {
					double[] centroid = roi.getContourCentroid();
					if (CentrTrackRoiUtils.isRoiTouchingEdge(roi, imagePlusDna))
						g.setColor(Color.RED);
					else
						g.setColor(Color.GREEN);
					int r = 10;
					g.fillOval((int) centroid[0] - r / 2, (int) centroid[1] - r / 2, r, r);

					g.setColor(Color.BLUE);
					AffineTransform scaleMatrix = new AffineTransform();
					scaleMatrix.translate(centroid[0], centroid[1]);
					scaleMatrix.scale(u1, u1);
					scaleMatrix.translate(-centroid[0], -centroid[1]);
					g.draw(scaleMatrix.createTransformedShape(roi.getPolygon()));

					g.setColor(Color.CYAN);
					scaleMatrix = new AffineTransform();
					scaleMatrix.translate(centroid[0], centroid[1]);
					scaleMatrix.scale(u2, u2);
					scaleMatrix.translate(-centroid[0], -centroid[1]);
					g.draw(scaleMatrix.createTransformedShape(roi.getPolygon()));
				}

				// Centrosomes
				for (Spot s : msi.centrosome_spot_list) {
					if (s.getFeature(Spot.FRAME) == f) {
						int r = 10;
						int px = (int) cax.rawValue(s.getFeature(Spot.POSITION_X).doubleValue());
						int py = (int) cay.rawValue(s.getFeature(Spot.POSITION_Y).doubleValue());

						if (s.getFeatures().containsKey("WhereInNuclei")) {
							if (s.getFeature("WhereInNuclei") == 0.0)
								g.setColor(Color.LIGHT_GRAY);
							if (s.getFeature("WhereInNuclei") == 1.0)
								g.setColor(Color.BLUE);
							if (s.getFeature("WhereInNuclei") == 2.0)
								g.setColor(Color.GREEN);
						} else
							g.setColor(Color.YELLOW);
						g.drawOval(px - r / 2, py - r / 2, r, r);

						g.setColor(Color.WHITE);
						if (s.getFeatures().containsKey("Centrosome"))
							g.drawString(String.valueOf("C" + s.getFeature("Centrosome").intValue()), px, py);

						r = 4;
						if (s.getFeatures().containsKey("Nuclei") && s.getFeatures().containsKey("NuclX")
								&& s.getFeatures().containsKey("NuclY")) {
							int nx = (int) cax.rawValue(s.getFeature("NuclX"));
							int ny = (int) cay.rawValue(s.getFeature("NuclY"));

							g.drawOval(nx - r / 2, ny - r / 2, r, r);
							g.drawString(String.valueOf(s.getFeature("Nuclei").intValue()), nx + 5, ny + 5);
						}
					}
				}
			}

			// draw time information
			g.setFont(new Font("Helvetica", Font.PLAIN, 16));
			g.drawString(String.valueOf(f), 10, height - 20);

			// new ImagePlus("Java 2D Demo", bi).show();

			// copy image to stack
			RandomAccess<ARGBType> raOut = imgOut.randomAccess();
			for (int w = 0; w < width; w++)
				for (int h = 0; h < height; h++) {
					raOut.setPosition(new int[] { w, h, 0, f });
					ARGBType t = raOut.get();
					int pix = bi.getRGB(w, h);
					t.set(pix);
				}
		}
		// ImageJFunctions.show(imgOut);
		return imgOut;
	}

	private Img<ARGBType> renderCellMarkers(Img<ARGBType> imgOut, int nucId) {
		int width = (int) imgOut.dimension(0);
		int height = (int) imgOut.dimension(1);
		long frames = imgOut.dimension(3);
		long[] min = new long[imgOut.numDimensions()];
		long[] max = new long[imgOut.numDimensions()];
		imgOut.min(min);
		imgOut.max(max);
		logService.debug(Arrays.toString(min) + " " + Arrays.toString(max));

		for (int f = 0; f < frames; f++) {
			long[] mini = new long[] { min[0], min[1], min[2], f };
			long[] maxi = new long[] { max[0], max[1], max[2], f };

			RandomAccessibleInterval<ARGBType> vwOut = Views.interval(imgOut, mini, maxi);
			ImagePlus imp = ImageJFunctions.wrap(vwOut, "");

			BufferedImage bi = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
			Graphics2D g = (Graphics2D) bi.getGraphics();
			g.drawImage(imp.getImage(), 0, 0, null);

			for (MasterSheetItem msi : ms.values()) {
				// ROIs
				Roi roi = CentrTrackRoiUtils.getRoiForFrame(msi.rois_in_the_same_track, f);
				if (roi != null) {
					// if (CentrTrackRoiUtils.isRoiTouchingEdge(roi,
					// imagePlusDna))
					// g.setColor(Color.DARK_GRAY);
					// else
					// g.setColor(Color.GRAY);
					// g.draw(roi.getPolygon());

					double[] centroid = roi.getContourCentroid();
					AffineTransform scaleMatrix = new AffineTransform();
					scaleMatrix.translate(centroid[0], centroid[1]);
					scaleMatrix.scale(u1, u1);
					scaleMatrix.translate(-centroid[0], -centroid[1]);
					g.setColor(Color.RED);
					g.draw(scaleMatrix.createTransformedShape(roi.getPolygon()));

					g.setColor(Color.BLUE);
					int r = 10;
					g.fillOval((int) centroid[0] - r / 2, (int) centroid[1] - r / 2, r, r);
				}

				// Centrosomes
				for (Spot s : msi.centrosome_spot_list) {
					if (s.getFeature(Spot.FRAME) == f) {
						int r = 10;
						int px = (int) cax.rawValue(s.getFeature(Spot.POSITION_X).doubleValue());
						int py = (int) cay.rawValue(s.getFeature(Spot.POSITION_Y).doubleValue());
						g.setColor(Color.YELLOW);
						g.fillOval(px - r / 2, py - r / 2, r, r);

						r = 4;
						g.setColor(Color.WHITE);
						if (s.getFeatures().containsKey("NuclX") && s.getFeatures().containsKey("NuclY")) {
							int nx = (int) cax.rawValue(s.getFeature("NuclX"));
							int ny = (int) cay.rawValue(s.getFeature("NuclY"));

							g.drawLine(nx, ny, px, py);
						}
					}
				}
			}

			// new ImagePlus("Java 2D Demo", bi).show();

			// copy image to stack
			RandomAccess<ARGBType> raOut = imgOut.randomAccess();
			for (int w = 0; w < width; w++)
				for (int h = 0; h < height; h++) {
					raOut.setPosition(new int[] { w, h, 0, f });
					ARGBType t = raOut.get();
					int pix = bi.getRGB(w, h);
					t.set(pix);
				}
		}
		// ImageJFunctions.show(imgOut);
		return imgOut;
	}

	public void renderResultTable() {
		ResultsTable rt = new ResultsTable();
		Analyzer.setResultsTable(rt);

		for (Iterator<SpotCollection> centrSC = spotCentrosomes.iterator(); centrSC.hasNext();) {
			for (Iterator<Spot> sIt = centrSC.next().iterator(true); sIt.hasNext();) {
				Spot s = sIt.next();
				rt.incrementCounter();
				if (s.getFeature("Nuclei") != null)
					rt.addValue("Nuclei", s.getFeature("Nuclei"));
				rt.addValue("Centrosome", s.getFeature("Centrosome"));
				rt.addValue("Frame", s.getFeature(Spot.FRAME));
				rt.addValue("WhereInNuclei", s.getFeature("WhereInNuclei"));
				rt.addValue("ValidCentroid", s.getFeature("ValidCentroid"));
				rt.addValue("CentX", s.getFeature(Spot.POSITION_X));
				rt.addValue("CentY", s.getFeature(Spot.POSITION_Y));
				rt.addValue("Time", s.getFeature(Spot.POSITION_T));
			}
		}
		rt.showRowNumbers(false);
		rt.save(path + "data/" + filename + "-table.csv");
		// rt.show("Results");

		rt = new ResultsTable();
		Analyzer.setResultsTable(rt);
		TrackModel tmn = trkMtModelNuclei.getTrackModel();
		for (Integer trkID : tmn.trackIDs(true)) {
			Set<Spot> nucleiSpots = tmn.trackSpots(trkID);
			for (Spot s : nucleiSpots) {
				rt.incrementCounter();
				rt.addValue("Nuclei", ms.get(trkID).final_id);
				rt.addValue("Frame", s.getFeature(Spot.FRAME));
				rt.addValue("NuclX", s.getFeature(Spot.POSITION_X));
				rt.addValue("NuclY", s.getFeature(Spot.POSITION_Y));
			}
		}
		rt.showRowNumbers(false);
		rt.save(path + "data/" + filename + "-nuclei.csv");
	}

	public static void main(String[] args) throws IOException {
		// create the ImageJ application context with all available services
		final ImageJ ij = new ImageJ();
//		ij.ui().setHeadless(true);
		 ij.ui().showUI();

		List<String> filesList = new ArrayList<String>();
		// add file values to list
//		filesList.add("../data/PC/input/centr-pc-0.tif");
//		filesList.add("../data/PC/input/centr-pc-1.tif");
//		filesList.add("../data/PC/input/centr-pc-3.tif");
//		filesList.add("../data/PC/input/centr-pc-4.tif");
//		filesList.add("../data/PC/input/centr-pc-5.tif");
//		filesList.add("../data/PC/input/centr-pc-10.tif");
//		filesList.add("../data/PC/input/centr-pc-12.tif");
//		filesList.add("../data/PC/input/centr-pc-14.tif");
//		filesList.add("../data/PC/input/centr-pc-17.tif");
//		filesList.add("../data/PC/input/centr-pc-18.tif");
//		filesList.add("../data/PC/input/centr-pc-200.tif");
//		filesList.add("../data/PC/input/centr-pc-201.tif");
//		filesList.add("../data/PC/input/centr-pc-202.tif");
//		filesList.add("../data/PC/input/centr-pc-203.tif");
//		filesList.add("../data/PC/input/centr-pc-204.tif");
//		filesList.add("../data/PC/input/centr-pc-205.tif");
//		filesList.add("../data/PC/input/centr-pc-207.tif");
//		filesList.add("../data/PC/input/centr-pc-209.tif");
//		filesList.add("../data/PC/input/centr-pc-210.tif");
//		filesList.add("../data/PC/input/centr-pc-211.tif");
//		filesList.add("../data/PC/input/centr-pc-212.tif");
//		filesList.add("../data/PC/input/centr-pc-213.tif");
//		filesList.add("../data/PC/input/centr-pc-214.tif");
//		filesList.add("../data/PC/input/centr-pc-216.tif");
//		filesList.add("../data/PC/input/centr-pc-218.tif");
//		filesList.add("../data/PC/input/centr-pc-219.tif");
//		filesList.add("../data/PC/input/centr-pc-220.tif");
//		filesList.add("../data/PC/input/centr-pc-221.tif");
//		filesList.add("../data/PC/input/centr-pc-222.tif");
//		filesList.add("../data/PC/input/centr-pc-223.tif");
//		filesList.add("../data/PC/input/centr-pc-224.tif");
//
//		filesList.add("../data/Dyn/input/centr-dyn-101.tif");
//		filesList.add("../data/Dyn/input/centr-dyn-102.tif");
//		filesList.add("../data/Dyn/input/centr-dyn-103.tif");
//		filesList.add("../data/Dyn/input/centr-dyn-104.tif");
//		filesList.add("../data/Dyn/input/centr-dyn-105.tif");
//		filesList.add("../data/Dyn/input/centr-dyn-107.tif");
//		filesList.add("../data/Dyn/input/centr-dyn-109.tif");
//		filesList.add("../data/Dyn/input/centr-dyn-110.tif");
//		filesList.add("../data/Dyn/input/centr-dyn-112.tif");
//		filesList.add("../data/Dyn/input/centr-dyn-203.tif");
//		filesList.add("../data/Dyn/input/centr-dyn-204.tif");
//		filesList.add("../data/Dyn/input/centr-dyn-205.tif");
//		filesList.add("../data/Dyn/input/centr-dyn-207.tif");
//		filesList.add("../data/Dyn/input/centr-dyn-208.tif");
//		filesList.add("../data/Dyn/input/centr-dyn-209.tif");
//		filesList.add("../data/Dyn/input/centr-dyn-210.tif");
//		filesList.add("../data/Dyn/input/centr-dyn-213.tif");
//
//		filesList.add("../data/DIC1/input/centr-dic1-201.tif");
//		filesList.add("../data/DIC1/input/centr-dic1-203.tif");
//		filesList.add("../data/DIC1/input/centr-dic1-204.tif");
//
//		filesList.add("../data/Dyn-Kin1/input/centr-dynkin1-000.tif");
//		filesList.add("../data/Dyn-Kin1/input/centr-dynkin1-001.tif");
//		filesList.add("../data/Dyn-Kin1/input/centr-dynkin1-002.tif");
//		filesList.add("../data/Dyn-Kin1/input/centr-dynkin1-004.tif");
//		filesList.add("../data/Dyn-Kin1/input/centr-dynkin1-008.tif");
//		filesList.add("../data/Dyn-Kin1/input/centr-dynkin1-009.tif");
//		filesList.add("../data/Dyn-Kin1/input/centr-dynkin1-010.tif");
//		filesList.add("../data/Dyn-Kin1/input/centr-dynkin1-014.tif");
//		filesList.add("../data/Dyn-Kin1/input/centr-dynkin1-015.tif");
//		filesList.add("../data/Dyn-Kin1/input/centr-dynkin1-016.tif");
//		filesList.add("../data/Dyn-Kin1/input/centr-dynkin1-018.tif");
//		filesList.add("../data/Dyn-Kin1/input/centr-dynkin1-019.tif");
//		filesList.add("../data/Dyn-Kin1/input/centr-dynkin1-020.tif");
//		filesList.add("../data/Dyn-Kin1/input/centr-dynkin1-021.tif");
//		filesList.add("../data/Dyn-Kin1/input/centr-dynkin1-022.tif");
//		filesList.add("../data/Dyn-Kin1/input/centr-dynkin1-023.tif");
//		filesList.add("../data/Dyn-Kin1/input/centr-dynkin1-100.tif");
//		filesList.add("../data/Dyn-Kin1/input/centr-dynkin1-102.tif");
//		filesList.add("../data/Dyn-Kin1/input/centr-dynkin1-103.tif");
//		filesList.add("../data/Dyn-Kin1/input/centr-dynkin1-104.tif");
//		filesList.add("../data/Dyn-Kin1/input/centr-dynkin1-105.tif");
//		filesList.add("../data/Dyn-Kin1/input/centr-dynkin1-106.tif");
//		filesList.add("../data/Dyn-Kin1/input/centr-dynkin1-108.tif");
//		filesList.add("../data/Dyn-Kin1/input/centr-dynkin1-109.tif");
//		filesList.add("../data/Dyn-Kin1/input/centr-dynkin1-111.tif");
//		filesList.add("../data/Dyn-Kin1/input/centr-dynkin1-116.tif");
//		filesList.add("../data/Dyn-Kin1/input/centr-dynkin1-117.tif");
//		filesList.add("../data/Dyn-Kin1/input/centr-dynkin1-119.tif");
//		filesList.add("../data/Dyn-Kin1/input/centr-dynkin1-120.tif");
//		filesList.add("../data/Dyn-Kin1/input/centr-dynkin1-122.tif");
//		filesList.add("../data/Dyn-Kin1/input/centr-dynkin1-123.tif");
//		filesList.add("../data/Dyn-Kin1/input/centr-dynkin1-124.tif");
//
//		filesList.add("../data/DynCDK1as/input/centr-dyncdk1as-002.tif");
//		filesList.add("../data/DynCDK1as/input/centr-dyncdk1as-003.tif");
//		filesList.add("../data/DynCDK1as/input/centr-dyncdk1as-005.tif");
//		filesList.add("../data/DynCDK1as/input/centr-dyncdk1as-007.tif");
//		filesList.add("../data/DynCDK1as/input/centr-dyncdk1as-008.tif");
//		filesList.add("../data/DynCDK1as/input/centr-dyncdk1as-011.tif");

//		filesList.add("../data/DynCDK1as/input/centr-kin1-100.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-103.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-104.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-105.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-112.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-113.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-115.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-119.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-120.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-124.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-126.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-128.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-129.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-130.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-131.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-200.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-201.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-204.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-206.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-207.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-208.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-209.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-210.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-211.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-212.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-213.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-214.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-215.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-217.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-218.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-219.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-220.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-221.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-223.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-226.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-228.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-229.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-230.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-231.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-232.tif");
//		filesList.add("../data/DynCDK1as/input/centr-kin1-233.tif");
		filesList.add("../data/DynCDK1as/input/centr-kin1-234.tif");

		for (String filename : filesList) {
			final File file = new File(filename);

			if (file != null) {
				// load the dataset
				ij.log().info("Processing " + file.getPath());
				final Dataset dataset = ij.scifio().datasetIO().open(file.getPath());

				// show the image
				// ij.ui().show(dataset);

				// --------------------------------
				// invoke the plugin
				// --------------------------------
				// populate the map of input parameters
				final Map<String, Object> inputMap = new HashMap<String, Object>();
				inputMap.put("currentData", dataset);
				inputMap.put("showRender", false);
				// execute asynchronously using the command service
				final Future<CommandModule> future = ij.command().run(CentrosomeTracking.class, true, inputMap);
				// wait for the execution thread to complete
				final Module module = ij.module().waitFor(future);
			}
		}

		ij.log().info("Process finished!");

	}
}

class MasterSheetItem {
	String nuclei_roi_id;
	List<String> rois_in_the_same_track;
	int first_frame;
	int final_id;
	int n_centrosomes;
	List<Spot> centrosome_spot_list;

	public static MasterSheetItem getMasterSheetItemFromFinalId(Iterator<MasterSheetItem> msl, int final_id) {
		while (msl.hasNext()) {
			MasterSheetItem msi = msl.next();
			// System.out.println("searching for sheet item with final value " +
			// final_id + " current " + msi.final_id);
			if (msi.final_id == final_id)
				return msi;
		}
		return null;
	}
}

class CentrTrackRoiUtils {
	public static Roi getRoiForFrame(List<String> rois_list, int frame) {
		RoiManager manager = RoiManager.getInstance();
		if (manager == null || rois_list == null)
			return null;

		Roi[] rois = manager.getRoisAsArray();
		// search for corresponding roi based on list
		for (String roi_id : rois_list) {
			// extract frame and corr id
			int roiframe = Integer.valueOf(roi_id.substring(0, 4));

			if (frame == roiframe) {
				for (Roi roi : rois) {
					if (roi.getName().equals(roi_id))
						return roi;
				}
			}
		}
		return null;
	}

	public static boolean isRoiTouchingEdge(Roi roi, ImagePlus img) {
		Rectangle r = roi.getBounds();
		if (r.x == 0 || r.y == 0 || r.x + r.width == img.getWidth() || r.y + r.height == img.getHeight())
			return true;
		else
			return false;
	}
}
