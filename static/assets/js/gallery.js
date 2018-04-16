/*
 * Copyright (c) 2018 - Dumi Loghin (dumi@makerlala.com)
 */
	
/*
 * Initialize
 */
function init() {
	// hide search panel
	var search = document.getElementById("search");
	if (search != null)  {
		search.style.display = "none";
	}
	// get window dimensions
	$.ajax({
		type: "POST",
		url : "/reportsize",
		data : {
			window_height : $(window).height(),
			window_width : $(window).width(),
			document_height : $(document).height(),
			document_width : $(document).width(),
		},
	});
}

/*
 * Toggle search panel
 */
function showSearch() {
	var search = document.getElementById("search");
	if (search.style.display === "none") {
		search.style.display = "block";
		document.getElementById("query").focus();
	} else {
		search.style.display = "none";
	}
}

/*
 * Go to index.html
 */
function goHome() {
	window.location = '/';
}

/*
 * Go to settings
 */
function showSettings() {
	window.location = '/settings';
}

/*
 * Go to AI page
 */
function showAi() {
	window.location = '/ai';
}

function startDetection() {
	$.ajax({
		type: "POST",
		url : "/detect",
		data : {},
	});
}

function startTraining() {
	$.ajax({
		type: "POST",
		url : "/train",
		data : {},
	});
}

