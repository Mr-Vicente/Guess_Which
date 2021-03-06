/* 
	Artcore Template
	http://www.templatemo.com/preview/templatemo_423_artcore
*/

jQuery(document).ready(function($) {

	'use strict';

	/************** Full Screen Slider *********************/
    	$(window).resize(function(){
    	    var height = $(window).height();
    	    var width  = $(window).width();
    	    $('.swiper-container, .swiper-slide').height(height);
    	    $('.swiper-container, .swiper-slide').width(width);

    	})
    	$(window).resize(); 


    	$('.arrow-left, .arrow-right').on('click', function() {
            $('.slider-caption h2').removeClass('animated fadeInDown');
            $('.slider-caption h2').fadeIn(0).addClass('animated fadeInDown');
            $('.slider-caption p').removeClass('animated fadeInUp');
            $('.slider-caption p').fadeIn(0).addClass('animated fadeInUp');
        });

    	var mySwiper = new Swiper('.swiper-container',{
    	    mode:'horizontal',
    	    loop: true,
    	    keyboardControl: true
    	  })
    	  $('.arrow-left').on('click', function(e){
    	    e.preventDefault()
    	    mySwiper.swipePrev()
    	  })
    	  $('.arrow-right').on('click', function(e){
    	    e.preventDefault()
    	    mySwiper.swipeNext()

    	})


    /************** Animated Hover Effects *********************/
        $('.staff-member').hover(function(){
    	    $('.overlay .social-network').addClass('animated fadeIn');
    	}, function(){
    	    $('.overlay .social-network').removeClass('animated fadeIn');
    	});

    	$('.blog-thumb, .project-item').hover(function(){
    	    $('.overlay-b a').addClass('animated fadeIn');
    	}, function(){
    	    $('.overlay-b a').removeClass('animated fadeIn');
    	});


    /************** FancyBox Lightbox *********************/
        $(".fancybox").fancybox();

});


    /************** Blog Masonry Isotope *********************/
        $(window).load(function () {
            blogisotope = function () {
                var e, t = $(".blog-masonry").width(),
                    n = Math.floor(t);
               	if ($(".blog-masonry").hasClass("masonry-true") === true) {
                    n = Math.floor(t * .3033);
                    e = Math.floor(t * .04);
                    if ($(window).width() < 1023) {
                        if ($(window).width() < 768) {
                            n = Math.floor(t * 1)
                        } else {
                            n = Math.floor(t * .48)
                        }
                    } else {
                        n = Math.floor(t * .3033)
                    }
                }
                return e
            };
            var r = $(".blog-masonry");
            bloggingisotope = function () {
                r.isotope({
                    itemSelector: ".post-masonry",
                    animationEngine: "jquery",
                    masonry: {
                        gutterWidth: blogisotope()
                    }
                })
            };
            bloggingisotope();
            $(window).smartresize(bloggingisotope)
        })
/************** OUR CODE after loading *********************/
$(document).ready(function() {
    var questionBox = document.getElementById("question")
    questionBox.addEventListener("keyup", function(event) {
  // Number 13 is the "Enter" key on the keyboard
        console.log(event.keyCode)
        if (event.keyCode === 13) {
            event.preventDefault();
            document.getElementById("ask_button").click();
        }
    });
});


/************** OUR CODE *********************/

//var images_to_guess = ["images/blog-1.jpg", "images/blog-2.jpg", "images/blog-3.jpg", "images/blog-4.jpg", "images/blog-5.jpg", "images/blog-6.jpg"];
//var chosen_image = "images/blog-1.jpg";

var images_to_guess = [];
var chosen_image;
var difficulty = 1;

var i_think_not = [];
for (var i = 0; i < images_to_guess.length; i++) {
    i_think_not.push(false);
}


function changeDifficulty(level){
    difficulty = level;
    var display = document.getElementById("apply_changes");
    display.innerHTML="Please press 'New Game' to apply changes";
}


async function displayImages() {
    if (images_to_guess.length == 0)
        await newGame(false);
    var display = document.getElementById("image_display");
    display.innerHTML="";
    for (var i in images_to_guess) {
        var div1 = document.createElement("div");
        div1.className = "overlay-b";
        div1.id = "overlay_".concat(i.toString());
        div1.innerHTML =
            `<div class="overlay-inner">
                 <a-yes id="yes_`+ i +`" class="fa fa-check" onclick="guessImage(this,`+ i +`)" ></a-yes>
                 <a-no class="fa fa-times" onclick="guessNot(this,`+ i +`)"></a-no>
             </div>`;


        var img = document.createElement("img");
        img.src = '../static/VQA_dataset/Images/mscoco/val2014/' + images_to_guess[i];

        var div3 = document.createElement("div");
        div3.className = "blog-thumb";
        div3.appendChild(img);
        div3.appendChild(div1);
            
        var div4 = document.createElement("div");
        div4.className = "post-masonry col-md-4 col-sm-6";
        div4.appendChild(div3);

        display.appendChild(div4);
    }

}

function guessImage(element, i) {

    if(i_think_not[i]){
        return;
    }

    var overlay = document.getElementById("overlay_".concat(i.toString()));

    overlay.style = `opacity: 1;
                    visibility: visible;
                    `;

    var modal = document.getElementById("pop-up");
    var title = document.getElementById("pop-up-title");
    var text = document.getElementById("pop-up-text");


    if (chosen_image === images_to_guess[i]) {
        //WON
        text.innerHTML = "Congratulations";
        title.innerHTML = "YOU WON!!!";
        element.style.color = "#34db2e";
    }
    else {
        //LOST
        text.innerHTML = "Sucker";
        title.innerHTML = "You Lost";
        element.style.color = "#db2e2e";
    }
    modal.style.display = "block";
    modal.style.backdrop = 'static';
    modal.style.keyboard = false;


}

function guessNot(element, i) {
    var overlay = document.getElementById("overlay_".concat(i.toString()));
    /*if (i_think_not[i]) {
        element.style.color = "white";
        overlay.style = `opacity: 0;
                    visibility: hidden;
                    `;
    } else {
         element.style.color = "#db2e2e";
                overlay.style = `opacity: 1;
                            visibility: visible;
                            `;
    }
    i_think_not[i] = !i_think_not[i];
    */
    element.style.color = "#db2e2e";
        overlay.style = `opacity: 1;
                    visibility: visible;
                    `;
    i_think_not[i] = true;
    var yes = document.getElementById("yes_".concat(i.toString()));
    yes.style.color = "white";
}


function _setDifficulty(level){
    var ele = document.getElementsByName('difficulty');
    for(i = 0; i < ele.length; i++) {
        if(ele[i].value == level)
            ele[i].checked = "checked";
    }
}


async function newGame(pressed) {

    var apply = document.getElementById("apply_changes");
    apply.innerHTML="";
    document.getElementById("imagesrc").src = "../static/bot_shrug.png"  //aqui ?? esta
    _setDifficulty(difficulty);

    await $.post("/start_game", {"difficulty": difficulty},
        function (data) {
            var response = data
            chosen_image = response.Chosen;
            
            images_to_guess = response.Indexes;
            var ri = randomNumber(0,images_to_guess.length)
            images_to_guess.splice(ri, 0,chosen_image)
            console.log(images_to_guess)
        }
    );
    if(pressed){
        await displayImages();
    }
    var modal = document.getElementById("pop-up");
    modal.style.display = "none";

}

function randomNumber(min, max) { 
    return Math.random() * (max - min) + min;
} 
var nGuesses = 0
function ask(input_question) {
    nGuesses+=1
    document.getElementById("answer").innerHTML = "...";
    $.post("/ask",
        { "id": chosen_image, "question": input_question },
        function (data) {
            var response = data
            var success = response.Success;
            /*if (success == false)
                showRIPIcon();*/
            var answer = response.Answer;
            var guess = response.Bot_guess;
            console.log(guess);
            document.getElementById("answer").innerHTML = answer;
            document.getElementById("imagesrc").src = '../static/VQA_dataset/Images/mscoco/val2014/'+guess   //?? esta que tens de mudar e v?? se o path esst?? certo

        }
    );
}

