import geopandas as gpd

class DataBuilder:

    # /// <summary>
    # /// Everything relies on a table containing the area key to zone code index lookup.
    # /// This is calculated from the shapefile, by going though all the areas and numbering them (from zero).
    # /// The number of areas found determines the order of the T and dis matrices.
    # /// There is a potential problem here if the trips file contains an area not in the shapefile. If we don't know the
    # /// location of an area then we have to forget about it.
    # /// This is static as all the other methods require this to be created first.
    # /// </summary>
    # /// <param name="Shpfilename"></param>
    # /// <returns>A data table containing the areakey to zone code lookup</returns>
    def DeriveAreakeyToZoneCodeFromShapefile(ShpFilenameWGS84):
        #read shapefile and create areakey, zonei, centroid and area table
        #DataTable dt = new DataTable("zones");
        #dt.Columns.Add("areakey", typeof(string));
        #dt.Columns.Add("zonei", typeof(int));
        #dt.Columns.Add("lat", typeof(float));
        #dt.Columns.Add("lon", typeof(float));
        #dt.Columns.Add("osgb36_east", typeof(float));
        #dt.Columns.Add("osgb36_north", typeof(float));
        #dt.Columns.Add("area", typeof(float));
        #dt.PrimaryKey = new DataColumn[] { dt.Columns[0] }; #set primary key or keys here

        dt = {} #primary key is areakey


        #go through shapefile and write unique zones to the table, along with their (lat/lon) centroids and areas
        UniqueAreaKeys = {}
        #load the shapefile
        features = gpd.read_file(ShpFilename)
        #string prj = ShapeUtils.GetPRJ(ShpFilename);
        #create a math transform to convert OSGB36 into WGS84
        #CoordinateSystem sourceCS = (CoordinateSystem)sf.CoordinateSystem;
        #CoordinateSystemFactory csFac = new ProjNet.CoordinateSystems.CoordinateSystemFactory();
        #CoordinateSystem sourceCS = (CoordinateSystem)csFac.CreateFromWkt(prj);
        #CoordinateSystem destCS = (CoordinateSystem)csFac.CreateFromWkt(wktWGS84);
        #CoordinateSystem destCS = GeographicCoordinateSystem.WGS84;
        #CoordinateTransformationFactory ctFac = new CoordinateTransformationFactory();
        #CoordinateTransformation ct = (CoordinateTransformation)ctFac.CreateFromCoordinateSystems(sourceCS, destCS); #THIS NEEDS TO BE LENIENT
        #now process the shapefile

        for idx, f in shapefile.iterrows():
            Geometry g = (Geometry)f.Geometry;
            double length = g.Length;
            Point centroid = (Point)g.Centroid;
            #now need the area key code
            #string AreaKey = fdr.ItemArray[1] as string;
            #there are only three columns in the shapefile, the_geom (=0), code (=1, MSOA11CA) and plain text name (=2, MSOA11NM)
            string AreaKey = f.Attributes["MSOA11CD"] as string;
            if (!UniqueAreaKeys.Contains(AreaKey))
            {
                DataRow row = dt.NewRow();
                row["areakey"] = AreaKey;
                row["zonei"] = UniqueAreaKeys.Count; //0 based index (original VB code was 1 based)
                #reproject centroid into WGS84
                double[] latLon = ct.MathTransform.Transform(new double[] { centroid.Coordinate.X, centroid.Coordinate.Y });
                row["lat"] = (float)latLon[1];
                row["lon"] = (float)latLon[0];
                row["osgb36_east"] = (float)centroid.Coordinate.X;
                row["osgb36_north"] = (float)centroid.Coordinate.Y;
                row["area"] = g.Area;
                dt.Rows.Add(row);
                UniqueAreaKeys.Add(AreaKey);
            }
            else
            {
                #else warn duplicate area? might happen if islands split, then have to be careful about centroids and areas
                System.Diagnostics.Debug.WriteLine("WARNING: Duplicate area: " + AreaKey);
            }
        #end for

        #save data table
        #using (Stream stream = File.Create("c:\\richard\\ZoneCodes.bin"))
        #{
        # BinaryFormatter serializer = new BinaryFormatter();
        # serializer.Serialize(stream, dt);
        #}
        #Serialiser.Put("c:\\richard\\ZoneCodes.bin", dt);
        System.Diagnostics.Debug.WriteLine("DeriveAreakeyToZoneCodeFromShapefile discovered " + dt.Rows.Count + " zones");
        return dt;
    
################################################################################