//
//  BubblyEffect.swift
//  RanKing
//
//  Created by Damario Hamilton on 11/29/25.
//

import SwiftUI

struct BubblyEffect: ViewModifier {
    @State private var scale: CGFloat = 1.0
    
    func body(content: Content) -> some View {
        content
            .scaleEffect(scale)
            .onAppear {
                withAnimation(
                    Animation.easeInOut(duration: 2.8)
                        .repeatForever(autoreverses: true)
                ) {
                    scale = 1.06
                }
            }
    }
}

extension View {
    func bubbly() -> some View {
        self.modifier(BubblyEffect())
    }
}

struct StaggeredBubbly: ViewModifier {
    var delay: Double
    @State private var scale: CGFloat = 1.0
    
    func body(content: Content) -> some View {
        content
            .scaleEffect(scale)
            .onAppear {
                withAnimation(
                    Animation.easeInOut(duration: 2.8)
                        .repeatForever(autoreverses: true)
                        .delay(delay)
                ) {
                    scale = 1.06
                }
            }
    }
}

extension View {
    func staggeredBubbly(_ delay: Double) -> some View {
        self.modifier(StaggeredBubbly(delay: delay))
    }
}

// Pulsing Dot on the login code

struct PulsingNotificationDot: View {
    @State private var scale: CGFloat = 1.0
    @State private var opacity: Double = 1.0
    
    var body: some View {
        ZStack {
            // Outer pulse glow
            Circle()
                .fill(Color.red.opacity(0.5))
                .frame(width: 16, height: 16)
                .scaleEffect(scale)
                .opacity(opacity)
            
            // Solid inner dot
            Circle()
                .fill(Color.red)
                .frame(width: 12, height: 12)
        }
        .onAppear {
            withAnimation(
                Animation.easeInOut(duration: 1.4)
                    .repeatForever(autoreverses: true)
            ) {
                scale = 1.8
                opacity = 0.2
            }
        }
    }
}

struct RankedLogoWithDot: View {
    var body: some View {
        ZStack(alignment: .topTrailing) {
            Image("appIcon")
                .resizable()
                .frame(width: 88, height: 88)
                .cornerRadius(22)
                .shadow(radius: 10, x: 0, y: 4)

            PulsingNotificationDot()
                .offset(x: 6, y: -6) // adjust dot position
        }
        .frame(width: 88, height: 88)
    }
}





