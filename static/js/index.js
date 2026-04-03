// Video carousel autoplay when in view (desktop) or play/pause based on visibility (mobile)
function setupVideoCarouselAutoplay() {
    const carouselVideos = document.querySelectorAll('.results-carousel video');
    
    if (carouselVideos.length === 0) return;
    
    const isMobile = window.matchMedia("(max-width: 768px)").matches;
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const video = entry.target;
            if (entry.isIntersecting && entry.intersectionRatio > 0.6) {
                // Video is sufficiently in view
                if (video.hasAttribute("autoplay") || !isMobile) {
                    // Desktop: autoplay if attribute exists
                    // Mobile: only play if user interacts (controls)
                    video.play().catch(e => {
                        // Autoplay failed, probably due to browser policy
                        console.log('Autoplay prevented:', e);
                    });
                }
            } else {
                // Video is out of view, pause it
                video.pause();
            }
        });
    }, {
        threshold: [0, 0.6, 1.0] // Multiple thresholds for better detection
    });
    
    carouselVideos.forEach(video => {
        observer.observe(video);
    });
}

$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    // Check if mobile
    const isMobile = window.matchMedia("(max-width: 768px)").matches;
    var carousels = [];

    // Only initialize bulma-carousel on desktop
    if (!isMobile) {
      var options = {
        slidesToScroll: 1,
        slidesToShow: 1,
        loop: true,
        infinite: true,
        autoplay: false, // Disable autoplay, we'll control it via video ended events
        autoplaySpeed: 5000,
      }

      // Initialize all div with carousel class (desktop only)
      carousels = bulmaCarousel.attach('.carousel', options);
      
      // Setup video-ended handler and manual navigation detection for each carousel
      carousels.forEach((carousel, carouselIndex) => {
        const carouselElement = carousel.element;
        const videos = carouselElement.querySelectorAll('video');
        let isManualNavigation = false;
        let currentVideoIndex = 0;
        
        // Track manual navigation (user clicks navigation buttons or dots)
        carouselElement.addEventListener('click', (e) => {
          // Check if user clicked on navigation arrows or dots
          if (e.target.closest('.carousel-navigation') || 
              e.target.closest('.carousel-dot') || 
              e.target.closest('[data-action]')) {
            isManualNavigation = true;
            // Pause current video if any
            const currentSlide = carouselElement.querySelectorAll('.item')[carousel.index];
            if (currentSlide) {
              const currentVideo = currentSlide.querySelector('video');
              if (currentVideo) {
                currentVideo.pause();
              }
            }
          }
        });
        
        // When slide changes, setup video ended handler
        carousel.on('show', (state) => {
          const currentSlide = carouselElement.querySelectorAll('.item')[state.index];
          if (currentSlide) {
            const video = currentSlide.querySelector('video');
            if (video) {
              // Define handler function
              const handleVideoEnd = () => {
                // Move to next slide when video ends
                carousel.next();
                // Reset manual navigation flag after video ends
                isManualNavigation = false;
              };
              
              // Remove previous ended listener if any (using named function reference)
              const previousHandler = video._carouselEndHandler;
              if (previousHandler) {
                video.removeEventListener('ended', previousHandler);
              }
              
              // Store handler reference for cleanup
              video._carouselEndHandler = handleVideoEnd;
              
              // Add new ended listener
              video.addEventListener('ended', handleVideoEnd, { once: true });
              
              // If video is already ended, move to next immediately
              if (video.ended) {
                handleVideoEnd();
              } else if (!isManualNavigation) {
                // If not manual navigation, play the video
                video.play().catch(e => {
                  console.log('Video play prevented:', e);
                });
              }
            }
          }
        });
        
        // Start first video when carousel is ready
        setTimeout(() => {
          const firstSlide = carouselElement.querySelectorAll('.item')[0];
          if (firstSlide) {
            const firstVideo = firstSlide.querySelector('video');
            if (firstVideo && !isManualNavigation) {
              firstVideo.play().catch(e => {
                console.log('First video play prevented:', e);
              });
            }
          }
        }, 100);
      });
    }

    // Mobile: disable autoplay for carousel videos and add controls
    if (isMobile) {
      document.querySelectorAll(".results-carousel video").forEach(v => {
        v.autoplay = false;
        v.loop = true;
        v.muted = true;
        v.playsInline = true;
        v.preload = "metadata";
        v.controls = true;
        v.removeAttribute("autoplay");
      });
    }
    
    // Setup video autoplay for carousel (desktop) or view-based play/pause (mobile)
    setupVideoCarouselAutoplay();

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();

})

// Video autoplay for details sections
document.addEventListener("toggle", (e) => {
  if (e.target.tagName !== "DETAILS" || !e.target.open) return;
  
  e.target.querySelectorAll("video").forEach(v => {
    v.muted = true;
    v.playsInline = true;
    v.setAttribute("autoplay", "");
    v.setAttribute("loop", "");
    
    // Ensure video is loaded before playing
    if (v.readyState >= 2) {
      const p = v.play();
      if (p && typeof p.catch === "function") p.catch(() => {});
    } else {
      v.load();
      v.addEventListener("loadeddata", function() {
        const p = v.play();
        if (p && typeof p.catch === "function") p.catch(() => {});
      }, { once: true });
    }
  });
}, true);

// Also handle videos that are already in open details on page load
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("details[open] video").forEach(v => {
    v.muted = true;
    v.playsInline = true;
    v.setAttribute("autoplay", "");
    v.setAttribute("loop", "");
    const p = v.play();
    if (p && typeof p.catch === "function") p.catch(() => {});
  });
});
