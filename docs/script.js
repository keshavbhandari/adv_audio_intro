var header = document.querySelector("header");
var stickyNav = document.querySelector("#stickyNav");
var title = document.querySelector("#title")

// TODO: throttle this function for optimal performance in production
window.addEventListener('scroll', function(e){
  var scrollPos = window.pageYOffset || document.documentElement.scrollTop;
  var stickyLine = header.scrollHeight - stickyNav.scrollHeight;
  if(scrollPos > stickyLine){
    stickyNav.classList.add("fixed");
    stickyLine.classList.remove("title");
  }else if(stickyNav.classList.contains('fixed')){
    stickyNav.classList.remove("fixed");
    stickyLine.classList.remove("title");
  }
});