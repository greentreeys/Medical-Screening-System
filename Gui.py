###################################################################
###################################################################
######################## imports ##################################
import gtk;
###################################################################
###################################################################


###################################################################
###################################################################
################# Input Dialog 1 ##################################






class InputDialog(gtk.Dialog):



    def Default_operation(self,checkbox,button):
	    if checkbox.get_active():
		    button.set_sensitive(False)
	    else:
		    button.set_sensitive(True)
    
    def Reset_mask(self,button):

	    self.button_vessel.set_sensitive(True);
	    self.file_vessel=None; 
	    self.label_vessel_File.set_text("No File")
	    self.checkbox_vessel_Default.set_active(False);

	    self.button_macula.set_sensitive(True);
	    self.file_macula=None; 
	    self.label_macula_File.set_text("No File")
	    self.checkbox_macula_Default.set_active(False);
	    
	    self.button_od.set_sensitive(True);
	    self.file_od=None; 
	    self.label_od_File.set_text("No File")
	    self.checkbox_od_Default.set_active(False);

    def Reset_gmp(self,button):
	    self.adjustment_npv.set_value(250);
	    self.adjustment_min_angle.set_value(-10);
	    self.adjustment_max_angle.set_value(10);
	    self.adjustment_step.set_value(4);

    def Reset_dark(self,button):
	
	    self.adjustment_cmin.set_value(250)
 	    self.adjustment_cmax.set_value(-10)
	    self.adjustment_dstep.set_value(10)
	    self.adjustment_dmax.set_value(10)
	    self.adjustment_dmin.set_value(10)
	    self.adjustment_nl.set_value(10)

	
    def Reset_bright(self,button):
	    self.adjustment_b_cmin.set_value(250)
 	    self.adjustment_b_cmax.set_value(-10)
	    self.adjustment_b_dstep.set_value(10)
	    self.adjustment_b_dmax.set_value(10)
	    self.adjustment_b_dmin.set_value(10)
	    self.adjustment_b_nl.set_value(10)
    
    def Get_form_values(self):
	    data={};
	    data['Vessel']=self.file_vessel;
	    data['Macula']=self.file_macula;
	    data['OD']=self.file_od;
	    data['GMP_Pivots']=self.adjustment_npv.get_value();
	    data['GMP_Min_Angle']=self.adjustment_min_angle.get_value();
	    data['GMP_Max_Angle']=self.adjustment_max_angle.get_value();
	    data['GMP_Step']=self.adjustment_step.get_value();
	    data['Dark_Center_Min']=self.adjustment_cmin.get_value()
 	    data['Dark_Center_Max']=self.adjustment_cmax.get_value()
	    data['Dark_Delta_Step']=self.adjustment_dstep.get_value()
	    data['Dark_Delta_Max']=self.adjustment_dmax.get_value()
	    data['Dark_Delta_Min']=self.adjustment_dmin.get_value()
	    data['Dark_Levels']=self.adjustment_nl.get_value()
	    data['Bright_Center_Min']=self.adjustment_b_cmin.get_value()
 	    data['Bright_Center_Max']=self.adjustment_b_cmax.get_value()
	    data['Bright_Delta_Step']=self.adjustment_b_dstep.get_value()
	    data['Bright_Delta_Max']=self.adjustment_b_dmax.get_value()
	    data['Bright_Delta_Min']=self.adjustment_b_dmin.get_value()
	    data['Bright_Levels']=self.adjustment_b_nl.get_value()

	    return data;





    def File_open(self,window,obj,file_var):
	   dialog = gtk.FileChooserDialog("Open..",None,gtk.FILE_CHOOSER_ACTION_OPEN,(gtk.STOCK_CANCEL,gtk.RESPONSE_CANCEL,gtk.STOCK_OPEN,gtk.RESPONSE_OK))
	   dialog.set_default_response(gtk.RESPONSE_OK)
	   response = dialog.run()
	   if response == gtk.RESPONSE_OK:
           	print dialog.get_filename(), 'selected'
	   	name=dialog.get_filename();
		file_var=name;
	   	if(len(name)>10):
			name=name[-10:];
			obj.set_text(name);

	   elif response == gtk.RESPONSE_CANCEL:
    		print 'Closed, no files selected'
	   elif response == -4:
    		print 'Closed, no files selected'
	   dialog.destroy()

    def __init__(self, parent=None):
        gtk.Dialog.__init__(self, "Size Groups", parent,gtk.DIALOG_MODAL,None);
        try:
            self.set_screen(parent.get_screen());
        except AttributeError:
            self.connect('destroy', lambda *w: gtk.main_quit());


	###################################################################
	###################################################################
	######################## MAIN VBOX ################################

	self.set_resizable(False);


########################################################################################################################################33
########################################################################################################################################33
################################################    ROW  1        ######################################################################33


        self.hbox_mini = gtk.HBox(False, 5);
        self.vbox.pack_start(self.hbox_mini, True, True, 0);
        
	###################################################################

        	
	self.vbox_level1 = gtk.VBox(False, 5)
	self.vbox_level1.set_border_width(5)
        self.hbox_mini.pack_start(self.vbox_level1, False, False, 0);
        self.size_group = gtk.SizeGroup(gtk.SIZE_GROUP_HORIZONTAL)
       

	###################################################################
	###################################################################
	######################## Frame MASK ###############################
       	
	self.frame_mask = gtk.Frame("Mask")
        self.vbox_level1.pack_start(self.frame_mask, True, True, 0)
	self.fixed_mask=gtk.Fixed();

	###################################################################
	###################################################################
	######################## Vessel MASK Buttons ######################

	self.button_vessel=gtk.Button("Choose File");
	self.file_vessel=None; ### Store the path of input file.

	self.button_vessel.show();
	self.label_vessel=gtk.Label("Vessel Mask");
	self.label_vessel.show();
	
	self.label_vessel_File=gtk.Label("No File");
	self.label_vessel_File.show();

	self.checkbox_vessel_Default=gtk.CheckButton("Default");
	self.checkbox_vessel_Default.show();	
	
	self.button_vessel.set_size_request(100,30);
	self.label_vessel.set_size_request(100,30);
	
	self.fixed_mask.put(self.label_vessel,20,15);
	self.fixed_mask.put(self.button_vessel,135,15);	
	self.fixed_mask.put(self.label_vessel_File,245,22);	
	self.fixed_mask.put(self.checkbox_vessel_Default,325,20);


	#################### Button Functionality #########################
	self.button_vessel.connect("clicked",self.File_open,self.label_vessel_File,self.file_vessel)
        self.checkbox_vessel_Default.connect("clicked", self.Default_operation,self.button_vessel)


	###################################################################
	###################################################################
	######################## Macula MASK Buttons ######################

	self.button_macula=gtk.Button("Choose File");
	self.button_macula.show();
	self.file_macula=None; ### Store the path of input file.
	
	self.label_macula=gtk.Label("Macula Mask");
	self.label_macula.show();
	
	self.label_macula_File=gtk.Label("No File");
	self.label_macula_File.show();

	self.checkbox_macula_Default=gtk.CheckButton("Default");	
	self.checkbox_macula_Default.show();	
	
	self.button_macula.set_size_request(100,30);
	self.label_macula.set_size_request(100,30);
	
	self.fixed_mask.put(self.label_macula,20,15+40);	
	self.fixed_mask.put(self.button_macula,135,15+40);	
	self.fixed_mask.put(self.label_macula_File,245,22+40);	
	self.fixed_mask.put(self.checkbox_macula_Default,325,20+40);	
	
	#################### Button Functionality #########################
	self.button_macula.connect("clicked",self.File_open,self.label_macula_File,self.file_macula)
        self.checkbox_macula_Default.connect("clicked", self.Default_operation,self.button_macula)

	
	###################################################################
	###################################################################
	######################## OD MASK Buttons ######################


	self.button_od=gtk.Button("Choose File")
	self.button_od.show()
	self.file_od=None; ### Store the path of input file.
	
	self.label_od=gtk.Label("Od Mask")
	self.label_od.show()
	
	self.label_od_File=gtk.Label("No File")
	self.label_od_File.show()

	self.checkbox_od_Default=gtk.CheckButton("Default")	
	self.checkbox_od_Default.show();	
	
	self.button_od.set_size_request(100,30)
	self.label_od.set_size_request(100,30)
	
	self.fixed_mask.put(self.label_od,20,15+40*2)	
	self.fixed_mask.put(self.button_od,135,15+40*2)	
	self.fixed_mask.put(self.label_od_File,245,22+40*2)	
	self.fixed_mask.put(self.checkbox_od_Default,325,20+40*2)	
	
	#################### Button Functionality #########################
	self.button_od.connect("clicked",self.File_open,self.label_od_File,self.file_od)
        self.checkbox_od_Default.connect("clicked", self.Default_operation,self.button_od)

	HIDE_MASK = gtk.Button("")
        HIDE_MASK.set_size_request(0, 0)
	HIDE_MASK.hide();
	self.fixed_mask.put(HIDE_MASK,425,20+150)	

	self.mask_reset = gtk.Button("Reset")
        self.mask_reset.set_size_request(80,30)
	self.fixed_mask.put(self.mask_reset,325,20+110)	
	
	self.mask_reset.connect("clicked",self.Reset_mask)

	self.frame_mask.add(self.fixed_mask)
	

########################################################################################################################################33
########################################################################################################################################33
###############################################   Level  2        ######################################################################33
########################################################################################################################################33

	
	###################################################################
	###################################################################
	######################## GMP Params ###############################

	self.vbox_level2 = gtk.VBox(False, 5)
        self.hbox_mini.pack_start(self.vbox_level2, False, False, 0)
	self.vbox_level2.set_border_width(5)

        self.frame_gmp = gtk.Frame("GMP Params")
        self.vbox_level2.pack_start(self.frame_gmp, True, True, 0)
	self.fixed_gmp=gtk.Fixed()


	###################################################################
	###################################################################
	######################## NPV Values ###############################
	
	
	self.label_npv=gtk.Label("NPV")
	self.label_npv.show()

	self.adjustment_npv =gtk.Adjustment(250,50,500,5,10,0)
	self.scale_npv=gtk.HScale(self.adjustment_npv)
	self.scale_npv.set_digits(0)
	self.scale_npv.set_size_request(300,50)

	self.fixed_gmp.put(self.label_npv,20,15)	
	self.fixed_gmp.put(self.scale_npv,120,-10)	
	
	###################################################################
	###################################################################
	######################## Min angle  ###############################



	self.label_min_angle=gtk.Label("Min Angle")
	self.label_min_angle.show()

	self.adjustment_min_angle =gtk.Adjustment(-10,-20,0,5,10,0)
	self.scale_min_angle=gtk.HScale(self.adjustment_min_angle)
	self.scale_min_angle.set_digits(0)
	self.scale_min_angle.set_size_request(300,50)

	self.fixed_gmp.put(self.label_min_angle,20,15 + 40)	
	self.fixed_gmp.put(self.scale_min_angle,120,-10+40)	


	###################################################################
	###################################################################
	######################## Step size  ###############################


	self.label_step=gtk.Label("Step size")
	self.label_step.show()

	self.adjustment_step =gtk.Adjustment(4,1,10,5,10,0)
	self.scale_step=gtk.HScale(self.adjustment_step)
	self.scale_step.set_digits(0)
	self.scale_step.set_size_request(300,50)

	self.fixed_gmp.put(self.label_step,20,15 + 80)	
	self.fixed_gmp.put(self.scale_step,120,-10+80)	

	###################################################################
	###################################################################
	######################## Max Angle  ###############################

	self.label_max_angle=gtk.Label("Max Angle")
	self.label_max_angle.show()

	self.adjustment_max_angle =gtk.Adjustment(10,0,20,5,10,0)
	self.scale_max_angle=gtk.HScale(self.adjustment_max_angle)
	self.scale_max_angle.set_digits(0)
	self.scale_max_angle.set_size_request(300,50)

	self.fixed_gmp.put(self.label_max_angle,20,15 + 120)	
	self.fixed_gmp.put(self.scale_max_angle,120,-10+120)	

	###################################################################
	###################################################################
	######################## Hide buttons  ############################

	HIDE_GMP = gtk.Button("")
        HIDE_GMP.set_size_request(0, 0)
	HIDE_GMP.hide();
	self.fixed_gmp.put(HIDE_GMP,425,20+170)	

	self.gmp_reset = gtk.Button("Reset")
        self.gmp_reset.set_size_request(80,30)
	self.fixed_gmp.put(self.gmp_reset,325,20+135)	
	self.frame_gmp.add(self.fixed_gmp)
	
	
	self.gmp_reset.connect("clicked",self.Reset_gmp)



########################################################################################################################################33
########################################################################################################################################33
################################################    ROW  2        ######################################################################33
########################################################################################################################################33


        self.hbox_mini_2 = gtk.HBox(False, 5)
        self.vbox.pack_start(self.hbox_mini_2, True, True, 0)
        
	
	###################################################################
	###################################################################
	######################## Dark Image Options  ######################

	self.vbox_level3 = gtk.VBox(False, 5)
        self.hbox_mini_2.pack_start(self.vbox_level3, False, False, 0)	
	self.vbox_level3.set_border_width(5)


        self.frame_dark = gtk.Frame("Dark Image")
        self.vbox_level3.pack_start(self.frame_dark, True, True, 0)
	self.fixed_dark=gtk.Fixed()


	###################################################################
	###################################################################
	######################## Center Min  ######################
	
	
	self.label_cmin=gtk.Label("Center Min")
	self.label_cmin.show()

	self.adjustment_cmin =gtk.Adjustment(250,50,500,5,10,0)
	self.scale_cmin=gtk.HScale(self.adjustment_cmin)
	self.scale_cmin.set_digits(0)
	self.scale_cmin.set_size_request(300,50)

	self.fixed_dark.put(self.label_cmin,20,15)	
	self.fixed_dark.put(self.scale_cmin,120,-10)	


	###################################################################
	###################################################################
	######################## Center Max  ######################

	self.label_cmax=gtk.Label("Center Max")
	self.label_cmax.show()

	self.adjustment_cmax =gtk.Adjustment(-10,-20,0,5,10,0)
	self.scale_cmax=gtk.HScale(self.adjustment_cmax)
	self.scale_cmax.set_digits(0)
	self.scale_cmax.set_size_request(300,50)

	self.fixed_dark.put(self.label_cmax,20,15 + 40)	
	self.fixed_dark.put(self.scale_cmax,120,-10+40)	

	###################################################################
	###################################################################
	######################## Delta Min  ######################

	self.label_dmin=gtk.Label("Delta Min")
	self.label_dmin.show()

	self.adjustment_dmin =gtk.Adjustment(4,1,10,5,10,0)
	self.scale_dmin=gtk.HScale(self.adjustment_dmin)
	self.scale_dmin.set_digits(0)
	self.scale_dmin.set_size_request(300,50)

	self.fixed_dark.put(self.label_dmin,20,15 + 80)	
	self.fixed_dark.put(self.scale_dmin,120,-10+80)	

	###################################################################
	###################################################################
	######################## Delta Step  ######################

	self.label_dstep=gtk.Label("Delta Step")
	self.label_dstep.show()

	self.adjustment_dstep =gtk.Adjustment(10,0,20,5,10,0)
	self.scale_dstep=gtk.HScale(self.adjustment_dstep)
	self.scale_dstep.set_digits(0)
	self.scale_dstep.set_size_request(300,50)

	self.fixed_dark.put(self.label_dstep,20,15 + 120)	
	self.fixed_dark.put(self.scale_dstep,120,-10+120)	

	###################################################################
	###################################################################
	######################## Delta Max  ######################

	
	self.label_dmax=gtk.Label("Delta MAx")
	self.label_dmax.show()

	self.adjustment_dmax =gtk.Adjustment(10,0,20,5,10,0)
	self.scale_dmax=gtk.HScale(self.adjustment_dmax)
	self.scale_dmax.set_digits(0)
	self.scale_dmax.set_size_request(300,50)

	self.fixed_dark.put(self.label_dmax,20,15 + 160)	
	self.fixed_dark.put(self.scale_dmax,120,-10+160)	

	##############################################################
	###################################################################
	###################################################################
	######################## Number of Levels  ######################

	
	self.label_nl=gtk.Label("Num Levels")
	self.label_nl.show()

	self.adjustment_nl =gtk.Adjustment(10,0,20,5,10,0)
	self.scale_nl=gtk.HScale(self.adjustment_nl)
	self.scale_nl.set_digits(0)
	self.scale_nl.set_size_request(300,50)

	self.fixed_dark.put(self.label_nl,20,15 + 200)	
	self.fixed_dark.put(self.scale_nl,120,-10+200)	


	#########################################################################################3333	
	
	HIDE_DARK = gtk.Button("")
        HIDE_DARK.set_size_request(0, 0)
	HIDE_DARK.hide();
	self.fixed_dark.put(HIDE_DARK,425,20+250)	

	self.dark_reset = gtk.Button("Reset")
        self.dark_reset.set_size_request(80,30)
	self.fixed_dark.put(self.dark_reset,325,20+215)	
	self.dark_reset.connect("clicked",self.Reset_dark)
	
	self.frame_dark.add(self.fixed_dark)



	###################################################################
	###################################################################
	######################## Bright Image##############################

	self.vbox_level4 = gtk.VBox(False, 5)
        self.hbox_mini_2.pack_start(self.vbox_level4, False, False, 0)
	self.vbox_level4.set_border_width(5)

        self.frame_bright = gtk.Frame("Bright Image")
        self.vbox_level4.pack_start(self.frame_bright, True, True, 0)
	self.fixed_bright=gtk.Fixed()

	########################################################################33
	###################################################################
	###################################################################
	######################## Bright Image Cmin##############################
	
	
	self.label_b_cmin=gtk.Label("Center Min")
	self.label_b_cmin.show()

	self.adjustment_b_cmin =gtk.Adjustment(250,50,500,5,10,0)
	self.scale_b_cmin=gtk.HScale(self.adjustment_b_cmin)
	self.scale_b_cmin.set_digits(0)
	self.scale_b_cmin.set_size_request(300,50)

	self.fixed_bright.put(self.label_b_cmin,20,15)	
	self.fixed_bright.put(self.scale_b_cmin,120,-10)	


	###################################################################
	###################################################################
	######################## Bright Image C Max ##############################

	self.label_b_cmax=gtk.Label("Center Max")
	self.label_b_cmax.show()

	self.adjustment_b_cmax =gtk.Adjustment(-10,-20,0,5,10,0)
	self.scale_b_cmax=gtk.HScale(self.adjustment_b_cmax)
	self.scale_b_cmax.set_digits(0)
	self.scale_b_cmax.set_size_request(300,50)

	self.fixed_bright.put(self.label_b_cmax,20,15 + 40)	
	self.fixed_bright.put(self.scale_b_cmax,120,-10+40)	



	###################################################################
	###################################################################
	######################## Bright Image D Min##############################

	self.label_b_dmin=gtk.Label("Delta Min")
	self.label_b_dmin.show()

	self.adjustment_b_dmin =gtk.Adjustment(4,1,10,5,10,0)
	self.scale_b_dmin=gtk.HScale(self.adjustment_b_dmin)
	self.scale_b_dmin.set_digits(0)
	self.scale_b_dmin.set_size_request(300,50)

	self.fixed_bright.put(self.label_b_dmin,20,15 + 80)	
	self.fixed_bright.put(self.scale_b_dmin,120,-10+80)	

	#######################################################p####################3
	###################################################################
	###################################################################
	######################## Bright Image D Step##############################

	self.label_b_dstep=gtk.Label("Delta Step")
	self.label_b_dstep.show()

	self.adjustment_b_dstep =gtk.Adjustment(10,0,20,5,10,0)
	self.scale_b_dstep=gtk.HScale(self.adjustment_b_dstep)
	self.scale_b_dstep.set_digits(0)
	self.scale_b_dstep.set_size_request(300,50)

	self.fixed_bright.put(self.label_b_dstep,20,15 + 120)	
	self.fixed_bright.put(self.scale_b_dstep,120,-10+120)	

	###################################################################
	###################################################################
	######################## Bright Image D Max##############################

	
	self.label_b_dmax=gtk.Label("Delta MAx")
	self.label_b_dmax.show()

	self.adjustment_b_dmax =gtk.Adjustment(10,0,20,5,10,0)
	self.scale_b_dmax=gtk.HScale(self.adjustment_b_dmax)
	self.scale_b_dmax.set_digits(0)
	self.scale_b_dmax.set_size_request(300,50)

	self.fixed_bright.put(self.label_b_dmax,20,15 + 160)	
	self.fixed_bright.put(self.scale_b_dmax,120,-10+160)	

	##############################################################
	###################################################################
	###################################################################
	######################## Bright Image NL ##############################

	
	self.label_b_nl=gtk.Label("Num Levels")
	self.label_b_nl.show()

	self.adjustment_b_nl =gtk.Adjustment(10,0,20,5,10,0)
	self.scale_b_nl=gtk.HScale(self.adjustment_b_nl)
	self.scale_b_nl.set_digits(0)
	self.scale_b_nl.set_size_request(300,50)

	self.fixed_bright.put(self.label_b_nl,20,15 + 200)	
	self.fixed_bright.put(self.scale_b_nl,120,-10+200)	
	
	HIDE_BRIGHT = gtk.Button("")
        HIDE_BRIGHT.set_size_request(0, 0)
	HIDE_BRIGHT.hide();
	self.fixed_bright.put(HIDE_BRIGHT,425,20+250)	

	self.bright_reset = gtk.Button("Reset")
        self.bright_reset.set_size_request(80,30)
	self.bright_reset.connect("clicked",self.Reset_bright)
	self.fixed_bright.put(self.bright_reset,325,20+215)	

	self.frame_bright.add(self.fixed_bright)


##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
        self.show_all()
##########################################################################################################################################
##########################################################################################################################################





   
#####################################################################################################33

##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################






class PyApp(gtk.Window):

    def __init__(self):
        super(PyApp, self).__init__()
        self.set_title("Simulation")
	self.set_default_size(800, 650)
	self.set_position(gtk.WIN_POS_CENTER)
	self.set_resizable(True)
        self.connect("destroy", gtk.main_quit)
	self.connect('check-resize',self. changed)
	scrolled_window = gtk.ScrolledWindow()
	scrolled_window.set_border_width(10)
	scrolled_window.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_ALWAYS)
	##### vars ##############################################################################################3
	self.fd_flag=-1;
	self.image_name=None; ### for single File
	self.directory_path=None; ## For Directory
	self.data={};
	self.results={};
	############ sample Results for display #######################
	
	self.results['attr1']="val1";
	self.results['attr2']="val2";
	self.results['attr3']="val3";
	self.results['attr4']="val4";
	self.results['attr5']="val5";
	self.results['attr6']="val6";
	self.results['attr7']="val7";
	self.results['attr8']="val8";
	self.results['attr9']="val9";
	self.results['attr10']="val10";
	self.results['attr11']="val11";
	self.results['attr12']="val12";
	self.results['attr13']="val13";


	################################################################
	##########################################################################################################
	#############################adding file in menu bar ####################################################
	mb = gtk.MenuBar()
	filemenu = gtk.Menu()	
	filem = gtk.MenuItem("File")
        filem.set_submenu(filemenu)
        mb.append(filem)
	############################ filemenu is the dropdown ###################################################
        sub_new = gtk.Menu()
        newfile = gtk.MenuItem("New")
        newfile.set_submenu(sub_new)
        newimage = gtk.MenuItem("Image")
	newimage.connect("activate", self.open_file)
        newdataset = gtk.MenuItem("Data Set")
        sub_new.append(newimage)
        sub_new.append(newdataset)
        filemenu.append(newfile)
	exit = gtk.MenuItem("Exit")
	save = gtk.MenuItem("Save")
	export = gtk.MenuItem("Export")
        exit.connect("activate", gtk.main_quit)
        filemenu.append(save)
        filemenu.append(export)
        filemenu.append(exit)
	######################################################################################################3####
	mainvbox=gtk.VBox();
	mainvbox.pack_start(mb, False, False, 0)
	hbox=gtk.HBox(False,10);	
	self.main_fixed=gtk.Fixed();
	self.fixed=gtk.Fixed();
    	self.fixed1=gtk.Fixed();
        vbox = gtk.VBox(False, 8)
	###########################################################################################################################
	####### Tool Bar Code ####################################################################################################
	##########################################################################################################################
	toolbar = gtk.Toolbar()
        toolbar.set_style(gtk.TOOLBAR_ICONS)
        newtb = gtk.ToolButton(gtk.STOCK_NEW)
        opentb = gtk.ToolButton(gtk.STOCK_OPEN)
        savetb = gtk.ToolButton(gtk.STOCK_SAVE)
        sep = gtk.SeparatorToolItem()
        quittb = gtk.ToolButton(gtk.STOCK_QUIT)
        toolbar.insert(newtb, 0)
        toolbar.insert(opentb, 1)
        toolbar.insert(savetb, 2)
        toolbar.insert(sep, 3)
        toolbar.insert(quittb, 4)
        quittb.connect("clicked", gtk.main_quit)
	newtb.connect("clicked", self.open_file)
	mainvbox.pack_start(toolbar, False, False, 0)
	############################################################333333
	####### Side Table  Code 1 ##################################################################
	#############################################################################################
	self.side_table_1 = gtk.Table(2, 2, False)
        self.side_table_1.set_col_spacings(10)
	self.side_table_1.set_row_spacings(10)
        self.sw_1 = gtk.ScrolledWindow()
        self.sw_1.set_shadow_type(gtk.SHADOW_ETCHED_IN)
        self.sw_1.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
        self.statusbar_1 = gtk.Statusbar()
        self.store_1 = self.create_model(self.data)
        self.treeView_1 = gtk.TreeView(self.store_1)
        self.treeView_1.connect("row-activated", self.on_activated,self.statusbar_1) #### function
        self.treeView_1.set_rules_hint(True)
        self.sw_1.add(self.treeView_1)
        self.create_columns(self.treeView_1,"Attributes","Value")    ######## function
	#### a button just to hide it ###############################
	close = gtk.Button("")
        close.set_size_request(300, 0)
	close.hide();
	############################################################
        vbox.pack_start(self.sw_1, True, True, 0)
	vbox.pack_start(self.statusbar_1, False, False, 0)
	vbox.pack_start(close, False, False, 0)
	############################################################
	####### Side Table  Code 2 ####################################################################################################
	##########################################################################################################################
	self.side_table_2 = gtk.Table(2, 2, False)
        self.side_table_2.set_col_spacings(10)
	self.side_table_2.set_row_spacings(10)
        self.sw_2 = gtk.ScrolledWindow()
        self.sw_2.set_shadow_type(gtk.SHADOW_ETCHED_IN)
        self.sw_2.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
        self.statusbar_2 = gtk.Statusbar()
        self.store_2 = self.create_model(self.results)
        self.treeView_2 = gtk.TreeView(self.store_2)
        self.treeView_2.connect("row-activated", self.on_activated,self.statusbar_2) #### function
        self.treeView_2.set_rules_hint(True)
        self.sw_2.add(self.treeView_2)
        self.create_columns(self.treeView_2,"Attributes","Value")    ######## function
	#### a button just to hide it ###############################
	close = gtk.Button("")
        close.set_size_request(300, 0)
	close.hide();
	############################################################
        vbox.pack_start(self.sw_2, True, True, 0)
	vbox.pack_start(self.statusbar_2, False, False, 0)
	vbox.pack_start(close, False, False, 0)
	hbox.add(vbox);
	############################################################
	

	self.central_table = gtk.Table(2, 2, False)
        self.central_table.set_col_spacings(10)
	self.central_table.set_row_spacings(10)
        self.central_table.set_row_spacing(1,2)
        
	self.image = gtk.Image();
	self.pixbuf = gtk.gdk.pixbuf_new_from_file("redrock.JPG")
	self.scaled_buf = self.pixbuf.scale_simple(550,500,gtk.gdk.INTERP_BILINEAR)
	self.image.set_from_pixbuf(self.scaled_buf)
        self.image_align = gtk.Alignment(0, 0, 0, 0)
        self.image_align.add(self.image)
        self.central_table.attach(self.image_align, 0, 4, 1, 2, gtk.FILL,gtk.FILL, 0, 0);

	self.checkbox_dark=gtk.CheckButton("Dark")	
	self.checkbox_align = gtk.Alignment(0, 0, 0, 0)
        self.checkbox_align.add(self.checkbox_dark)
        self.central_table.attach(self.checkbox_align, 1, 2, 3, 4, gtk.FILL,gtk.FILL | gtk.EXPAND, 1, 1)


	self.checkbox_save=gtk.CheckButton("Save Results")	
	self.save_align = gtk.Alignment(0, 0, 0, 0)
        self.save_align.add(self.checkbox_save)
        self.central_table.attach(self.save_align, 2, 3, 3, 4, gtk.FILL,gtk.FILL | gtk.EXPAND, 1, 1)

       	self.checkbox_toggle=gtk.CheckButton("Toggle Image")	
	self.toggle_align = gtk.Alignment(0, 0, 0, 0)
        self.toggle_align.add(self.checkbox_toggle)
        self.central_table.attach(self.toggle_align, 3, 4, 3, 4, gtk.FILL,gtk.FILL | gtk.EXPAND, 1, 1)


        self.comment_label = gtk.Label("Comments")
        self.comment_align = gtk.Alignment(0, 0, 0, 0)
        self.comment_align.add(self.comment_label)
        self.central_table.attach(self.comment_align, 0, 1, 5,6, gtk.FILL,gtk.FILL | gtk.EXPAND, 1, 1)
            
       
	self.comment_text = gtk.Entry()
	self.comment_text.set_size_request(400, 30)
	self.comment_t_align = gtk.Alignment(0, 0, 0, 0)
        self.comment_t_align.add(self.comment_text)
        self.central_table.attach(self.comment_t_align, 1, 8, 5,6, gtk.FILL,gtk.FILL | gtk.EXPAND, 1, 1)
       

	self.simulate = gtk.Button("Start Simulation")
	self.simulate.set_size_request(120, 30)
        self.sim_align = gtk.Alignment(0, 0, 0, 0)
        self.sim_align.add(self.simulate)
        self.central_table.attach(self.sim_align, 1, 2, 7,8, gtk.FILL,gtk.FILL | gtk.EXPAND, 1, 1)
            
#########################################################################################333


###################333 Place 12 #######################################################################3
        self.compare = gtk.Button("Compare")
	self.compare.set_size_request(120, 30)
        self.comp_align = gtk.Alignment(0, 0, 0, 0)
        self.comp_align.add(self.compare)
        self.central_table.attach(self.comp_align, 2,3, 7,8, gtk.FILL,gtk.FILL | gtk.EXPAND, 1, 1)
######################################################################################################
        self.fixed.put(self.central_table,0,0)
	hbox.add(self.fixed);
	mainvbox.add(scrolled_window)
	scrolled_window.add_with_viewport(hbox)
	hbox.show()
	scrolled_window.show()
	self.add(mainvbox)
        self.show_all()

    def create_model(self,data):
        store = gtk.ListStore(str,str)
	keys=data.keys();
	for key in keys:
            store.append([key,data[key]])

        return store

    def create_columns(self, treeView,text1,text2):
    
        rendererText = gtk.CellRendererText()
        column = gtk.TreeViewColumn(text1, rendererText, text=0)
        column.set_sort_column_id(0)    
        treeView.append_column(column)
        
        rendererText = gtk.CellRendererText()
        column = gtk.TreeViewColumn(text2, rendererText, text=1)
        column.set_sort_column_id(1)    
        treeView.append_column(column)	
	

    def on_activated(self, widget, row, col,obj):
        model = widget.get_model()
	if(model[row][1]!=None):
		text = model[row][0] + "  :  " + model[row][1]
	else:
		text = model[row][0] + "  :  " + "None"
        obj.push(0, text)
    
    def changed(self,window):
	size=window.get_size();
	w=min(size);
	w=max(w-300,500);
	self.scaled_buf = self.pixbuf.scale_simple(w,w,gtk.gdk.INTERP_BILINEAR)
	self.image.set_from_pixbuf(self.scaled_buf)
	print window.get_size();

    def open_file(self,item):

	dialog = gtk.FileChooserDialog("Open..",None,gtk.FILE_CHOOSER_ACTION_OPEN,(gtk.STOCK_CANCEL,gtk.RESPONSE_CANCEL,gtk.STOCK_OPEN,gtk.RESPONSE_OK))
	dialog.set_default_response(gtk.RESPONSE_OK)
	response = dialog.run()
	if response == gtk.RESPONSE_OK:
		print dialog.get_filename(), 'selected'
		self.pixbuf = gtk.gdk.pixbuf_new_from_file(dialog.get_filename())
		self.scaled_buf = self.pixbuf.scale_simple(400,400,gtk.gdk.INTERP_BILINEAR)
		self.image.set_from_pixbuf(self.scaled_buf)
		self.fd_flag=0;
		self.image_name=dialog.get_filename();
	elif response == gtk.RESPONSE_CANCEL:
    		print 'Closed, no files selected'
	dialog.destroy()

	dialog=InputDialog(self);
	ok_button = dialog.add_button(gtk.STOCK_OK, gtk.RESPONSE_ACCEPT)
	cancel_button = dialog.add_button(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL)
	while(True):	
		response=dialog.run();
		print "ok"
		print response;
		if response == gtk.RESPONSE_ACCEPT:
			print 'selected'
			self.data=dialog.Get_form_values();
			self.store_1.clear();
			self.store_1 = self.create_model(self.data)
        		self.treeView_1.set_model(self.store_1)
			self.treeView_1.show();
			##################### DATA ######################

			### here if statement is to check the form validation #####
			#### As of now form is set to true ############3

			###################################################
			if(True):
				dialog.destroy();
				break;
		elif response == gtk.RESPONSE_CANCEL:
    			print 'Closed, no files selected'
			dialog.destroy();
			break;
		elif response == -4:
    			print 'Closed, no files selected'
			dialog.destroy();
			break;

	
	

        
       
PyApp()
gtk.main()
